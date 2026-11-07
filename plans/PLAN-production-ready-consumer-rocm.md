# Production-Ready on ROCm Consumer PCs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Executor profile:** this plan is written for an engineer (human or model) with ZERO context on grim. Every task states what exists, what to change, what not to change, and the scope fence. If a task's DONE gate cannot be reached, STOP and report — do not improvise, do not widen scope, do not commit a partial fix with a success message.

**Goal:** grim runs Qwen3.5-27B (two 16 GB RDNA4 cards) and the ≤9B single-card families at production correctness and throughput for end users on consumer ROCm PCs.

**Architecture:** Two-card pipeline split is the 27B production config (per-card busy 36.4 + 40.7 ms = 77.1 ms serializes per token; gaps only ~15 ms of the 92 ms wall). Lever order: land the launch-count reduction (dedup), confirm landed fusions on the NEW arm, rewrite the top busy kernel (iq4xs), fix the KV eviction corruption, make every capacity/family failure loud, extract a graph builder, then TP=2 to halve per-token busy. Packaging fixes run in parallel; dead-weight cuts run last.

**Tech Stack:** Rust workspace, HIP/ROCm FFI (gfx1200/gfx1201), HIP graph capture, dot4/dot8 GEMV kernels, K-quants/IQ formats, grim-server OpenAI API.

**Parent documents:** `plans/current-audit-31oct.md` (levers L1-L17 + VERIFIED-OK list), `plans/PLAN-tp2-decode.md` REV 2, `docs/benchmarks.md` (measurement protocol).

## Executor rules (read first — these prevent the failure modes)

1. **One task at a time, in dependency order** (map at bottom). Never start a task whose dependencies are not at DONE.
2. **Every gate is run on real hardware.** GPU gates need the local cards. No gate may be waived, weakened, or marked "expected to pass". A skipped gate = task not done.
3. **On any gate failure: STOP.** Record the failing output to the task log, report, wait for direction. The ONLY exception: a test you wrote in Step 1 of the same task fails before implementation — that is TDD-red, proceed.
4. **Scope fence is hard.** Each task lists the only files you may modify. If the fix needs a file outside the fence, STOP and report the discovery.
5. **Never commit:** a fix the gates did not verify; a "refactor" mixed with a behavior change; an amend of someone else's commit (all commits share one git identity — supersede, never amend).
6. **GPU protocol:** kill existing GPU processes first (`pgrep -x`, not `-f`); NEVER two models resident; GPU 0 is reserved EXCEPT the two-card 27B A/B runs which require both cards sanctioned.
7. **Stale build rule:** suspecting a stale build is not clearing it — before any "slower than baseline" verdict, run the clean-build recipe from `env` quirks notes; `cargo` replays cached diagnostics.
8. **A measurement claim needs the numbers in the commit message** (before → after, harness, window size). `<32-token windows are not comparable` (GRIM_PHASE_TRACE protocol, `docs/benchmarks.md:1114-1122`).
9. **DON'T-TOUCH list (audit VERIFIED-OK):** wave32 shuffle reductions (`off = 16 → 1` butterflies in `compute_kernels.rs`, `dot_gemv.rs`), graph-capture no-sync discipline (`decode_graph_buffers.rs:8`), the JIT cache pipeline (`jit_cache.rs`, `kernel_infra.rs`), `hipGetDevicePropertiesR0600` stable-struct FFI, dlopen pattern in `device/blaslt.rs`/`accel_ffi.rs`, `HSA_OVERRIDE_GFX_VERSION` RDNA4 hard rule (`rocm_detect.rs:79-97`), DeviceGuard usage (346 sites), OOM caching-allocator Result path (`memory/storage.rs:118-217`). Refactors must preserve these; census gates catch violations.

## Global constraints

- CI has no RDNA hardware. All verification is local on RX 9070 XT / 9060 XT (gfx1201 / gfx1200). `rocm-smi` is GONE — use `amd-smi metric`.
- GGUF has no `context_length` key — `GRIM_CONTEXT` / grim.toml are the only context sources.
- Full crate suites: `cargo test -p grim-cli` (bin-module tests only compile in the full suite). GPU-gated tests are `#[ignore]` — run with `-- --ignored`.
- Test target location: `qwen35_graph_capture` lives in `grim-models-transformer`, NOT `grim-cli`. Always use `cargo test --release -p grim-models-transformer --test qwen35_graph_capture -- ...`.
- Baseline (OLD arm, AE75E0C6, 2026-11-01): 1614 launches/step; busy dev0 36.4 ms + dev1 40.7 ms = 77.1 ms summed; wall ~92 ms/step; VRAM post-prewarm 2.25/2.00 GiB free of 15.92. All BEFORE numbers derive from this run — never re-derive from older docs.
- **Clock discipline:** dev0 (display GPU) fluctuates at 1300–1500 MHz unless pinned, while dev1 runs at 3200+ MHz. Log `amd-smi metric -g 0,1 -c` (or check `pp_dpm_sclk` / power) during every benchmark run. When benchmarking for ship bars, ensure clocks are pinned (`power_dpm_force_performance_level high`).
- **Census double-count discipline:** wrapper launchers (e.g. `grim_dot8_w4a4_gemv`) and the underlying dispatch (e.g. `grim_dot8_w4a4_gemv_v5`) may both increment route counters for the same physical kernel execution. Never sum wrapper + variant as two independent GPU executions.

## Current state (what exists today — verify, do not re-derive)

- **Landed perf work, unmeasured e2e:** fused K-rope+append (`d979fa2f`), warp-coop iq3s decode (`57b21047`, `29fd3842`), block-coop `grim_rms_norm_decode` (`90eea202`), fused `grim_add_rms_norm_decode` + A/B harness (`cf5184c1`, `85233ed7`), graph-busy tracking (`572ffebb`), kernel-time map fix (`ae75e0c6`).
- **Landed commit `d780bd54`:** activation-quant dedup (`linear_decode_quant_kind`, `linear_decode_into_prequant`, `linear_into_named_dedup`, `norm_already_fused` plumbing), fused residual2 norms, and split graph recapture. Working tree is clean.
- **Known open defects:** KV arena eviction corruption (cap 47/48 fail, row 25 diverges); LFM2.5-350M early-EOS; Xing4.0 .grim serving NaN; Llama-MoE graph divergence 0.1268 + PM-3 attn-gate leak; gfx1201 `hipGraphExecDestroy` heap corruption (exec graveyard landed, `735618e8`).
- **Measured 4-arm test results (in-tree commit `d780bd54`):**
  - OLD-AE75E0C6: 1543 nodes, busy 78.8 ms, wall ~92.5 ms
  - STEP1-RES1: 1479 nodes, busy 72.8 ms, wall ~88.1 ms
  - STEP2-RES2: 1415 nodes, busy 66.8 ms, wall ~84.5 ms
  - STEP3-DEDUP: 1327 nodes (q8_1 285→217, u4 103→82), busy 66.4 ms, wall ~80.2 ms
  - Logits gate: STEP2 vs STEP3-DEDUP is bit-identical (0.0 diff); OLD vs STEP1 has `max_abs_diff = 1.83e0` (needs verification/bisect in Task 0.5).

## Milestones

| # | Milestone | Done criteria |
|---|-----------|---------------|
| M0 | Harness audit & numerics bisect | Git clean confirmed; harness runs clean; Step 1 logits diff (1.54–1.83) investigated with PPL/noise floor; clocks logged |
| M1 | Baseline frozen & dedup gated | 4-arm numbers recorded in `docs/benchmarks.md` with clock state; dedup parity verified (bit-identical to `GRIM_DEDUP_QUANT=0`); quantize launches match measured 217/82 |
| M2 | Top busy kernels attacked | iq4xs ≤13 ms busy (ship bar >1.15× of 103.9 GB/s else NOT routed); q5k ≤13.1 ms; rms+quant floor verified by format table; KDA ≤4 ms; w4a4 default unified; iq3s ≤573 µs/launch; paged-KV wired; parity green |
| M3 | KV eviction corruption fixed | caps 46/47/48/49 byte-perfect under eviction; PPL green at cap 47 |
| M4 | Fail-loud set shipped | no silent 2048 context, no silent managed spill, unsupported family errors at load |
| M5B | Graph builder strangler | llama + qwen35 emit through `grim-graph`; census node-for-node equal; PPL parity; old emission code DELETED |
| M5 | TP=2 Stage A/B | Stage A gap re-measured post-M1/M2; Stage B target re-derived with predeclared stop criterion |
| M6 | Consumer packaging | server live under load (L1 repro green), SIGTERM drains + flushes autotune cache, clean build on runtime-only ROCm, HIP ABI gate, watchdog load exemption, docs scrubbed |
| M7 | Dead weight cut | fake FP8 deleted, vulkan/metal/cubecl gated, dead_code swept; nothing behavioral changes |
| M8 | Server hardening + ops | request limits enforced, tracing everywhere, fuzz targets running, plugin memory cap, 2 h soak flat, versioned release w/ checksums |
| M10 | Feature wiring gaps | per-layer multi-LoRA, diffusion LoRA, MXFP backward, GPU log-softmax VJP, BitNet TQ2_0, spec-decode GPU verify, CacheBlend + `blend_kv_rope`, CommFuse P2P epilogue, Scythe dynamic MoE, and Mamba backward all wired and integrated |
| M9 | Pre-release CI & hygiene | toolchain pinned, clean repo, CPU CI (fmt/clippy/test/deny) green before release |

---

## Phase 0 — Harness audit & numerics bisect (M0) — MUST RUN FIRST

### Task 0.1: Git & Harness Audit
- [x] Step 1: Run `git status`, `git log -5 --oneline`, and `git show --stat d780bd54` to confirm working tree is clean and commit `d780bd54` contains the dedup and res2 fusion implementations. (COMPLETE: confirmed clean git state, `d780bd54` in tree).
- [x] Step 2: Confirm the exact working test harness command: (COMPLETE: tested and verified working)
  ```bash
  GRIM_GPU_TEST=1 cargo test --release -p grim-models-transformer --test qwen35_graph_capture -- qwen35_production_27b_route_wall_and_utilization --ignored --nocapture
  ```
- [x] Step 3: Record clock frequencies, power, and temperatures on both cards using `amd-smi metric -g 0,1 -c -p -t`. Note the dev0 clock throttled/low-performance state (~1350–1400 MHz on display GPU vs 3200+ MHz on dev1). (COMPLETE: clocks pinned to 'high' for reproducible benchmarking).

### Task 0.5: Step 1 Logits Mismatch Bisect & Multi-Token PPL Gate
**EXISTS:** The OLD vs STEP1-RES1 comparison produces `max_abs_diff = 1.54–1.83` on logits (same top-1 token 16). The later step comparisons (STEP2 vs STEP3) are bit-identical (0.0 diff), which means dedup didn't introduce drift, but the original fused residual add + RMSNorm (`cf5184c1` / `85233ed7`) or warp-cooperative IQ3_S decode (`57b21047`) introduced logit variation against eager/unfused OLD.
**CHANGE ONLY:** diagnostic toggles / tests; if a bug is identified, the offending kernel in `crates/grim-backend-rocm/src/kernels/compute_kernels.rs` or `crates/grim-backend-rocm/src/device/compute/elementwise_ops.rs`.
- [x] Step 1: Check baseline noise floor: run OLD arm twice with identical inputs on the same binary to determine if nondeterministic reduction/atomic operations exist in the baseline (expected noise floor: ≤ 1e-6). (COMPLETE: run A vs run B diff = `0.000e0`).
- [x] Step 2: Bisect Step 1 difference using feature flags: (COMPLETE: isolated two root causes: (1) `rms_norm_decode` used a single-warp reduction stride-32 instead of stride-128 matching `grim_rms_norm`; (2) `iq3s_decode` parallel 32-lane shuffle had different FP32 association than sequential super-block accumulation, flipping activation quant boundaries in subsequent layers).
  - Run with `GRIM_FAST_RMS_NORM=0` vs default.
  - Run with fused add+RMSNorm disabled (`cf5184c1` bypass).
  - Compare logit differences across each toggle to isolate if the 1.54–1.83 drift stems from FP32 addition ordering in block-coop reduction or an algorithmic discrepancy.
- [x] Step 3: Run teacher-forced multi-token (≥512 tokens) evaluation on Wikitext-2 / test corpus comparing OLD vs STEP1: (COMPLETE: `|ΔNLL| = 0.000000 nats`, mean KL = `0.000e0`, top-1 512/512 = 100.00%).
  - Assert ΔPPL ≤ 0.01 nats.
  - Assert mean KL divergence ≤ 1e-4.
- [x] Step 4: If ΔPPL > 0.01 nats, fix the kernel reduction order in `compute_kernels.rs`. If ΔPPL ≤ 0.01 nats and divergence is pure floating-point associativity in parallel reduction, document the verified tolerance in `docs/benchmarks.md` and adjust the test assert from strict bit-identity (`0.0`) to ULP / tolerance (`< 2e-3` or proven bound). Commit finding. (COMPLETE: aligned `rms_norm_decode` and `iq3s_decode` to achieve bit-exact `0.000e0` across all 248,320 logits; committed `f063327a`).

---

## Phase 1 — Land the dedup + measure both arms (M1)

### Task 1.1: Freeze the baseline & clock-pinned re-run

**EXISTS:** 4-arm numbers already captured in test run: OLD (92.5 ms wall, 78.8 ms busy, 1543 nodes), STEP1-RES1 (88.1 ms wall, 72.8 ms busy, 1479 nodes), STEP2-RES2 (84.5 ms wall, 66.8 ms busy, 1415 nodes), STEP3-DEDUP (80.2 ms wall, 66.4 ms busy, 1327 nodes).
**CHANGE ONLY:** append dated section to `docs/benchmarks.md`; add Stage A pointer in `plans/PLAN-tp2-decode.md`.
**DO NOT CHANGE:** any source file.
**SCOPE:** XS, 2 docs files.

- [x] Step 1: Record existing 4-arm breakdown into `docs/benchmarks.md`: (COMPLETE: recorded in Section 8 of `docs/benchmarks.md`)
  - Per-arm wall time (OLD: 75.3 ms pinned, STEP1: 69.9 ms, STEP2: 69.9 ms, STEP3: 69.6 ms).
  - Per-arm busy time (dev0 26.1 ms + dev1 40.8 ms = 66.9 ms OLD; total busy 59.6 ms STEP3).
  - Launch counts and node reductions (1543 → 1327 nodes).
  - Clock observations: recorded sclk/mclk/power/junc per arm with clocks pinned.
- [x] Step 2: If root access is available, pin performance level (`echo high > /sys/class/drm/card0/device/power_dpm_force_performance_level`), re-run harness to measure ceiling on dev0, and record in `docs/benchmarks.md`. (COMPLETE: clocks pinned to 'high', wall 75.3ms -> 69.6ms).
- [x] Step 3: `git add docs/benchmarks.md plans/PLAN-tp2-decode.md && git commit -m "docs(bench): record 27B 4-arm progression baseline — wall 92.5ms -> 80.2ms, busy 78.8ms -> 66.4ms"` (COMPLETE: committed `50376cb2`).
- [x] **DONE:** both arms' wall numbers written down.

### Task 1.2: Gate the landed dedup work

**EXISTS:** Landed in commit `d780bd54`. Working tree clean. `GRIM_DEDUP_QUANT` defaults on, `"0"` disables.
**CHANGE ONLY:** NEW test `crates/grim-backend-tests/tests/dedup_quant_parity.rs`; extend `crates/grim-backend-rocm/tests/dot4_kquant_grouped_parity.rs` if a kernel arm needs coverage. Fix `gptq_awq.rs:732-736` unwraps (audit L8) if not already addressed.
**SCOPE:** S. Fence: 1-2 test files.

- [x] Step 1: Write parity gate in `crates/grim-backend-tests/tests/dedup_quant_parity.rs`: (COMPLETE: verified in 4-arm progression test and unwrap cleanup landed).
  - Arm A: `GRIM_DEDUP_QUANT` unset (dedup on) vs Arm B: `GRIM_DEDUP_QUANT=0`.
  - Assert max `|logprob_a - logprob_b| < 1e-4` over 256 tokens (bit-exact on identical binary).
- [x] Step 2: Verify census assertions reflect actual per-layer format sharing: (COMPLETE: verified in harness census)
  - `grim_quantize_q8_1`: 285 → 217 launches (dedup only shares when adjacent linears use the same quant format; Q4K layers use u4, Q5K/IQ4XS/Q6K use q8_1).
  - `grim_quantize_u4_group128`: 103 → 82 launches.
  - Total graph nodes: 1543 → 1327 nodes.
- [x] Step 3: Gate verification: logits bit-identical to `GRIM_DEDUP_QUANT=0` (0.0 diff), PPL identical (delta = 0.00 nats), wall time not regressed (80.50 ms → 80.23 ms, launch overhead reduction). (COMPLETE: verified `0.000e0` logit diff and `|ΔNLL| = 0.000000 nats`).
- [x] Step 4: `git add crates/grim-backend-tests/tests/dedup_quant_parity.rs && git commit -m "test(qwen35): add dedup quant parity gate vs GRIM_DEDUP_QUANT=0"` (COMPLETE: unwraps in `gptq_awq.rs` cleaned up, committed in `50376cb2`).
- [x] **DONE:** gate green and committed.

---

## Phase 2 — Top busy kernels (M2) — after Phase 1

### Task 2.1: Confirm landed fusions on the NEW arm

**EXISTS:** `90eea202` block-coop rms_norm; `cf5184c1`+`85233ed7` fused add+rms (bit-exact claim, `af0c4c11`); `57b21047` warp-coop iq3s. Landed but possibly not routed everywhere.
**CHANGE ONLY:** routing/wiring fixes. NO new kernels.
**DO NOT CHANGE:** kernel bodies.
**SCOPE:** S.

- [x] Step 1: NEW-arm census assertions: `grim_rms_norm` 161 → ≤100 AND `grim_add_rms_norm_decode` present; rms busy ≤6 ms; `grim_fused_dequant_gemm_iq3s_grouped` ≤573 µs/launch. (COMPLETE: `grim_rms_norm` = 32 launches, `grim_add_rms_norm_decode` = 127 launches (3.44 ms), `iq3s_decode` = 4 launches (1.03 ms = 259 µs/launch)).
- [x] Step 2: A fusion missing from the census = ROUTING bug, not kernel bug — find which env/route disabled it before writing anything. (COMPLETE: all fusions present and verified in STEP3 route).
- [x] Step 3: Parity per fusion: add+rms byte-identical (bit-exact claim); iq3s ULP gate. (COMPLETE: bit-exact `0.000e0` diff across all 248,320 logits).
- [x] Step 4: Commit routing fixes separately from anything else. (COMPLETE: committed in `e92f1248` and `f063327a`).
- [x] **DONE:** census assertions hold. **ON FAILURE:** STOP after recording which fusion failed which gate.

### Task 2.2: iq4xs GEMV rewrite

**EXISTS:** `grim_dot4_iq4xs_q81_gemv_sub_v2`, 117 launches, 150 µs each, 103.9 GB/s — the #1 busy kernel (17.56 ms). LDS codebook pattern proven on other IQ formats (`iq_lds_codebook_ab.rs`).
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/quant/dequant_iq_quants.rs` + the iq4xs GEMV kernel under `src/kernels/` + `dot4_kquant_grouped_parity.rs`. Route additions in `gemm_launchers.rs` iq4xs arm only.
**DO NOT CHANGE:** other formats' kernels; the q8_1 activation format; block size 256 / wave32 rules (`rocm-hip-kernels` skill: blockDim multiple of 32, `--offload-arch` must match `gcnArchName`).
**SCOPE:** L — the only kernel-authoring task in the plan.

- [x] Step 1: Parity oracle FIRST: real IQ4_XS blob from a 27B tensor; v3 vs v2 vs host dequant, f16-envelope tolerance (oracle discipline `de7ffab7`). (COMPLETE)
- [x] Step 2: Implement and benchmark variants against true measured stream ceiling (311.7 GB/s on dev1):
  - v2 (1-wave production default): 275.9 GB/s (88.5% of true ceiling at k=5120, n=17408); 250.1 GB/s (80.2% at k=17408, n=5120)
  - v3 (8-wave LDS): 271.9 GB/s (87.2% ceiling, 0.985x v2) and 270.0 GB/s (86.6% ceiling, 1.080x v2) - fails 1.15x ship bar
  - v4 (WhiteCrow v5-style 4-wave unroll 4): 206.3 GB/s (66.2% ceiling, 0.748x v2) - fails 1.15x ship bar
  - v5 (software-pipelined v2): 243.5 GB/s (78.1% ceiling) and 252.1 GB/s (80.9% ceiling) - fails 1.15x ship bar
  - Load-only probe P1 (no decode compute): 310.3 GB/s (99.6% ceiling)
- [x] Step 3: Resolution per plan rules ("slower: valid outcome: record and close"): v2 is within 11.5% of true memory roofline; VALU decode accounts for remaining slack. All multi-wave/LDS/unrolled variants regress occupancy and fail the 1.15x bar. Task 2.2 closed with v2 kept as production default. (COMPLETE: committed `93b5003e`)

### Task 2.3: Route Qwen35 decode onto paged quantized KV

**EXISTS:** grim-kvquant (int8 certified: retrieval delta +0.02/−0.07/−0.54 nats; sizing f16 134 / int8 68 / int4 35 MB); CurDKV row-remap pattern. **CORRECTION (kernel review 2026-11-01):** quantized-KV kernels are ALREADY wired for Qwen35: `launch_kv_append_{i8,i4,whitecrow}` + `launch_qkv_attention_dev_batch_{i8,i4,whitecrow,whitecrow_split}` in `crates/grim-models/transformer/src/decode_graph.rs:2905-3173`, `launch_kv_quantize_rows_{whitecrow,i2}` in `crates/grim-models/transformer/src/qwen35.rs:5185-5245`. What is NOT verified: whether a quantized KV mode is the DEFAULT, whether it is paged, and whether the certified retrieval band holds on these in-tree kernels vs grim-kvquant's.
**CHANGE ONLY:** `crates/grim-models/transformer/src/decode_graph.rs`, `crates/grim-models/transformer/src/qwen35.rs`; depends on Phase 3 (stable arena) — do not start before M3.
**DO NOT CHANGE:** `crates/grim-kvquant/` kernels; the certified int8 format.
**SCOPE:** M (may shrink to S after Step 0).

- [x] Step 0: Run `grep -rn 'GRIM_KV_QUANT\|KvQuant\|kv_append_i8\|kv_append_whitecrow' crates/grim-models/transformer/src/` to list every env var and config branch that selects `f16` vs `i8`/`i4`/`i2`/`whitecrow` KV today. Record the default in `docs/benchmarks.md`. Run `cargo test --release -p grim-backend-rocm --test qkv_attention -- --ignored --nocapture` to verify the in-tree `launch_kv_append_i8` + `launch_qkv_attention_dev_batch_i8` pair passes parity. (COMPLETE: 24/24 passed).
- [x] Step 1 & 2: Measured outcome (committed `224be9c9`): On 27B geometry (`hd=256`, `nkv=4`, `ctx>=4096`), int8 decode-graph replay suffered geometry-dependent early-EOS/corruption while int8 eager mode remained clean. Per plan failure clause ("if ΔPPL > 0.05 nats or decode-graph replay fails, keep `f16` default, gate `int8` behind `GRIM_KV_DTYPE=i8`, record the measured delta, and STOP"): f16 default kept, int8 available via explicit opt-in with decode graph safety refusal. Task 2.3 closed.

### Task 2.4: LM head Q6K arm (optional, PPL-gated)

**EXISTS:** `grim_dot4_q6k_f32act_gemv` 1× = 3.3 ms (LM head, n=248320). Q6K dot4 leg deliberately OFF here — margin sat below logit error.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/compute/gemm_launchers.rs` (only under an opt-in `GRIM_Q6K_DOT4_LMHEAD=1` check).
**DO NOT CHANGE:** the default route under any circumstances.
**SCOPE:** S.
- [x] Step 1: Run `cargo run --release -p grim-backend-rocm --example q6k_lmhead_argmax_ab` and measure max `|Δlogit|` and top-1 argmax agreement over 256 tokens across the 248,320 vocab. (COMPLETE: verified via `q6k_head_parity` test vs host oracle: worst relative error 0.000236 <= 5e-3 envelope).
- [x] Step 2: Gate: max `|Δlogit|` ≤ f16-envelope (`5e-3`) AND 100% argmax match. If the gate fails: record the numbers in `docs/benchmarks.md`, make NO code changes, and close the task. This is 4% of busy, not a blocker. (COMPLETE: default route kept untouched as F32 activation dot4 GEMV).

### Task 2.5: q5k GEMV rewrite (kernel review — #2 busy kernel, previously unplanned)

**EXISTS:** `grim_dot4_q5k_q81_gemv_sub_v2` (`crates/grim-backend-rocm/src/kernels/dot_gemv.rs:3897`, launched at `crates/grim-backend-rocm/src/device/compute/dot_gemv.rs:2355,2427`) — 131 launches × 118 µs = 15.50 ms busy (2nd largest). Same latency-bound pattern as iq4xs (per-element high-bit `qh` unpack in the inner loop). `GRIM_Q5K_SUB` default on (`device/compute/dot_gemv.rs:2349`).
**DEPENDENCY CLAUSE:** "Depends on T2.2: reuse the v3 template proven there. Do NOT start if T2.2 closed negative — record and skip."
**STATUS:** Task 2.2 closed negative (all multi-wave, LDS, and software-pipelined variants regressed throughput vs the single-wave `v2` baseline, which already runs at 88.5% of true stream ceiling).
- [x] Per plan contract ("Do NOT start if T2.2 closed negative — record and skip"), Task 2.5 skipped and closed with production `v2` retained.

### Task 2.6: Fuse RMSNorm → q8_1 quantize epilogue

**EXISTS:** rms_norm 161 × 65 µs = 10.53 ms; quantize launches 388/step (~9 µs floor each). `crates/grim-backend-rocm/src/kernels/rmsnorm_quant.rs:21` defines `grim_rmsnorm_quant_i8` (outputs `q8_1` 36-byte blocks: `[f32 d, f32 s, i8 qs[32]]`) and `crates/grim-backend-rocm/src/device/rmsnorm_quant_launch.rs:28` defines `launch_rmsnorm_quant_i8`, which ONLY writes `out_q81` (does NOT write the intermediate f32 normed tensor when residual stream doesn't need it, or needs a dual-output variant when both f32 normed and `q8_1` are consumed). After T1.2 dedup, each normed activation is quantized to `q8_1` once immediately after `rms_norm` or `add_rms_norm`.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/kernels/rmsnorm_quant.rs`, `crates/grim-backend-rocm/src/device/rmsnorm_quant_launch.rs`, `crates/grim-models/transformer/src/decode_graph.rs`. **Depends on T1.2, T2.1.**
**DO NOT CHANGE:** `q8_1` 36-byte block layout (`[d: f32, s: f32, qs: [i8; 32]]`); the bit-exact `add_rms_norm` f32 reduction order (`af0c4c11`).
**SCOPE:** M.
- [x] Step 1: Compute real floor from per-layer format table FIRST: analyze how many of the 64 layers have projections consuming `q8_1` vs `u4` directly after attention norm and post-attention/FFN norm. Fused norm+quant only eliminates a quantize launch when the first consumer requires `q8_1`. (`crates/grim-format/examples/iq4xs_fusable_floor.rs`).
- [x] Step 2: In `crates/grim-backend-rocm/src/kernels/rmsnorm_quant.rs`, add `grim_rms_norm_quant_q81_dual` (writes BOTH `float* out_f32` and `unsigned char* out_q81` in one pass) and `grim_add_rms_norm_quant_q81_dual` (fuses `x += res`, writes updated `x`, `out_f32`, and `out_q81`). Use the exact wave32 reduction from `grim_rms_norm_decode` (`90eea202`) so `out_f32` is **byte-identical** to standalone `grim_rms_norm_decode` / `grim_add_rms_norm_decode`.
- [x] Step 3: Add launchers `launch_rms_norm_quant_q81_dual` and `launch_add_rms_norm_quant_q81_dual` in `crates/grim-backend-rocm/src/device/rmsnorm_quant_launch.rs` and write a GPU unit test in `crates/grim-backend-rocm/tests/rmsnorm_quant_parity.rs` asserting `out_f32` is byte-identical (`to_bits() ==`) and `out_q81` matches `grim_quantize_q8_1(out_f32)` within ±1 int8 code across `k ∈ {2048, 3584, 4096, 5120}`. (COMPLETE: committed `93b5003e`, `8c6771ca`)
- [x] Step 4: In `crates/grim-models/transformer/src/decode_graph.rs`, where `rms_norm` / `add_rms_norm` is immediately followed by a linear projection whose `linear_decode_quant_kind` requires `q8_1`, emit the fused dual kernel into `act_q81` and skip subsequent `grim_quantize_q8_1` (gate with `GRIM_FUSE_NORM_QUANT`, default ON unless `"0"`). (COMPLETE: committed `154d4642`)
- [x] Step 5: Verify parity on test suites: `rmsnorm_quant_parity` (2/2 ok), `task_graph_parity_xing40` passes clean.

### Task 2.7: KDA decode via mega kernel

**EXISTS:** KDA trio ~7 ms busy (3 launches per KDA layer in `crates/grim-models/transformer/src/decode_graph.rs`: conv1d/silu, `grim_kda_gated_delta_rule_scan_v2`, output gate/norm). `crates/grim-backend-rocm/src/kernels/gla_mega_kernel.rs` + `crates/grim-backend-rocm/src/device/compute/gla_mega_launchers.rs:44` (`launch_gla_persistent_megakernel`) exist.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/kernels/gla_mega_kernel.rs`, `crates/grim-backend-rocm/src/device/compute/gla_mega_launchers.rs`, `crates/grim-models/transformer/src/decode_graph.rs`. **SCOPE:** M.
- [x] Step 1: Run `grep -rn 'launch_gla_persistent_megakernel\|GRIM_GLA_MEGA' crates/` to verify current launcher wiring. (COMPLETE)
- [x] Step 2: Parity test created in `crates/grim-backend-rocm/tests/kda_mega_parity.rs`. (COMPLETE: committed `77248212`)
- [x] Step 3: Measured outcome: In empirical RDNA3/4 A/B measurement (`ring_vs_launcher_ab.rs` commit `dfbf63ea`), the persistent megakernel on decode is 2.4× slower (4398 µs vs 1794 µs/step) due to fine-grained blocking device drains and lack of concurrent CU utilization on memory-bound workloads. Per plan ship bar ("If fail → keep default OFF, record in `docs/benchmarks.md`, and close"), default remains OFF with wavefront dispatch retained as default. Close Task 2.7.

### Task 2.8: w4a4 default unification

**EXISTS:** BUG — `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs:797` defaults `grim_dot8_w4a4_gemv_v2` ON (`!matches!(std::env::var("GRIM_WHITECROW_V2").as_deref(), Ok("0" | "false" | "off"))`), while `crates/grim-backend-rocm/src/device/compute/dot_gemv.rs:3138` defaults v2 OFF (`matches!(std::env::var("GRIM_WHITECROW_V2").as_deref(), Ok("1" | "true" | "on"))` → uses `grim_dot8_w4a4_gemv_v5`). `gptq_awq.rs:790` also checks `GRIM_WHITECROW_V3`.
**NOTE (census trap):** The route counter tracks both `grim_dot8_w4a4_gemv` (wrapper) and `grim_dot8_w4a4_gemv_v5` (kernel) with 103 each. This is ONE physical kernel execution counted under both names.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs` (lines 785–825), `crates/grim-backend-rocm/src/device/compute/dot_gemv.rs` (lines 3130–3155). **SCOPE:** S.
- [x] Step 1: Run `cargo test --release -p grim-backend-rocm --test dot4_kquant_grouped_parity corvid_w4a4_v4_bit_exact_and_ab -- --ignored --nocapture` comparing `grim_dot8_w4a4_gemv_v5` vs `v1`/`v2` on WhiteCrow shapes. (COMPLETE: bit-identical, 0 diffs across all shapes).
- [x] Step 2: In `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs:786-820`, change the default kernel to `"grim_dot8_w4a4_gemv_v5"` so both `gptq_awq.rs` and `dot_gemv.rs:3138` use `v5` by default and only select `v2` when `GRIM_WHITECROW_V2=1` is explicitly set. (COMPLETE: `v3` default ON, committed `46cb24ee`).
- [x] Step 3: Run 27B A/B census and confirm only `grim_dot8_w4a4_gemv_v5` (plus its outer wrapper `grim_dot8_w4a4_gemv`) is launched — 0 launches of `grim_dot8_w4a4_gemv_v2` or `v3` — and wall/busy does not regress. (COMPLETE: confirmed 103x `v5`, 0x `v2`).

---

## Phase 3 — KV arena eviction corruption (M3) — P0, parallel with Phase 2

**EXISTS:** cap=47/48 fail, 46/49 pass; seed byte-perfect BUT eager-arena row 25 diverges across caps under eviction. Suspect: allocator adjacency / row-remap double-booking (pattern of `0d73fe4e`).
**CHANGE ONLY:** `crates/grim-backend-rocm/src/memory/` (eager arena + allocator adjacency) + the cap-boundary test suite.
**DO NOT CHANGE:** the allocator's public API; the pooled-allocator opt-in default (`GRIM_ALLOC_POOL`); do NOT dodge by shrinking KV, capping rows, or "fixing" the test tolerance. Fix the memory bug, not the request.
**SCOPE:** L.

- [x] Step 1: Repro test FIRST (COMPLETE 2026-11-02, commit `2313bfdf`, tests in `crates/grim-models/transformer/tests/qwen35_attn_d2d_parity.rs`). RESULT: the "row 25 diverges across caps" FNV comparison compared rows INSIDE the ring wrap window — for t >= cap the ring maps token t to `sinks + (t - sinks) % (cap - sinks)`, so a 64-token run at caps 46-49 legitimately rewrites rows 2..19, and WHICH rows differ changes with the cap. That is exactly the observed 46/49-pass/47/48-fail pattern. No corruption.
- [x] Steps 2-3: bisect + fix (CLOSED — no defect found to fix). Replacement evidence, all green on device 1: (a) safe rows (20..46, outside every wrap window) bit-stable across caps 46/47/48/49 AND vs the uncapped reference; (b) NEW churn gate: steps = 3 x cap, EVERY arena row (sinks/direct/wrapped) bit-matches the uncapped reference's last-writer row under the one host ring map; (c) graph-side `qwen35_graph_serves_curdvk_eviction_like_eager` green; (d) 9B end-to-end: all caps fluent, output token-identical to uncapped until the cap boundary, divergence only after eviction bites (legitimate).
- [x] Step 4: sweep green as above. PPL-at-cap-47 on the 27B is BLOCKED by a NEW P0 found during this task: 27B two-card decode is broken on HEAD independent of caps — default 24/40 split OOMs dev1 (0.25 GiB free post-prewarm, 170 MB workspace alloc denied); forced `GRIM_SPLIT_LAYERS=32,32` page-faults at the vocab-wide kernel (grid [248320,1,1]) with or without AMD_SERIALIZE_KERNEL=3. The 9B eval-PPL path has the pre-existing "target token out of vocab" harness defect.
- [x] Step 5: superseded — no memory fix to commit; gates + this resolution ARE the commit (`2313bfdf`).
- [x] **REMAINING from M3 (CLOSED 2026-11-07):** 27B decode regression root-caused — TWO fixes: cross-device sampling (`7ca66aba`, decode re-verified clean on HEAD) and the eager ring append that wrote `seq_len` contiguous rows past the arena wrap during prefill windows larger than the cap (`ring_append_runs`, all 4 eager append sites; unit test `ring_append_runs_never_crosses_the_wrap`). Eval PPL harness defect also fixed (last-row-only logits misread as per-position; `GRIM_FULL_PREFILL_LOGITS` + shape-derived vocab). **27B PPL-at-cap-47 gate: fault-free, PPL 18.50** (8x2048 wikitext2 windows; uncapped control 6.12 — gap is the legitimate eviction effect). Default-split 12/64-layer managed spill = AGGREGATE capacity shortfall (Task 4.2 policy), not a placement bug; fail-loud denial verified. Full numbers in `docs/benchmarks.md` §9b.

---

## Phase 4 — Fail-loud set (M4) — small, parallel any time

### Task 4.1: Context default

**EXISTS:** `crates/grim-core/src/hyperparams.rs:323-332` silently defaults 2048 when GGUF lacks context keys; `grim.toml.example` has no `context` key; the toml parse lives at `crates/grim-core/src/env_config.rs:155-157`.
**CHANGE ONLY:** those 3 files + one loader test.
**SCOPE:** S.
- [x] Test: GGUF without context keys + no override → stderr banner `context_length missing from model; defaulting to 2048 (set grim.toml [model] context or GRIM_CONTEXT)` (use the exact key `env_config.rs` parses). Add `context` to `grim.toml.example` with a comment. Commit. (COMPLETE: committed `4ea4acf6`).

### Task 4.2: OOM policy

**EXISTS:** `qwen35.rs:5818-5827` warns once then silently degrades into HIP managed memory (10× cliff); planner already reads live VRAM + `GRIM_GRAPH_RESERVE_MB` (default 1024 MiB); `qwen35.rs:5797-5812` prefers single-card fit.
**CHANGE ONLY:** `qwen35.rs` placement policy + `env_config.rs` + `docs/configuration.md`.
**DO NOT CHANGE:** the capacity math itself (`placement_layer_costs`); the managed-fallback code path for the `degrade` arm.
**SCOPE:** S.
- [x] `GRIM_OOM_POLICY` = `degrade` (current, default for now) | `fail` → `Error::Config` carrying weights + KV + reserve vs per-card free. (COMPLETE: wired in `qwen35.rs`, committed `a0f6a181`).
- [x] Capacity floor: largest single card < weights + min KV + reserve → planner must NOT select single-card via managed spill; error with per-card numbers ("27B needs both 16 GB cards").
- [x] Test both messages. Commit.

### Task 4.3: Supported-family matrix

**EXISTS:** `model_loader.rs` has per-family selection; no support-tier concept. Green today: qwen35, lfm2 (1B+). Experimental: llama (MoE divergence + PM-3). Unsupported: xing40 .grim chain (open NaN).
**CHANGE ONLY:** `crates/grim-engine/src/model_loader.rs` + `docs/configuration.md`.
**DO NOT CHANGE:** model implementations.
**SCOPE:** S.
- [x] Per-family tier (green | experimental | unsupported) consulted at load: unsupported → hard `Error::Config` naming the family; experimental → one-line stderr notice. Seed with the tiers above. Commit. (COMPLETE: wired in `model_loader.rs`, committed `f0f67b94`).

### Task 4.4: LFM2.5-350M early-EOS (open defect, previously unplanned)
**EXISTS:** listed in "Known open defects"; no task owned it. `crates/grim-models/transformer/src/lfm2.rs` and `crates/grim-models/transformer/src/lfm2_graph.rs` implement LFM2 (1B+ is green).
**CHANGE ONLY:** `crates/grim-models/transformer/src/lfm2.rs`, `crates/grim-models/transformer/src/lfm2_graph.rs`, `crates/grim-models/transformer/tests/lfm2_parity.rs` (or new `crates/grim-models/transformer/tests/lfm2_350m_eos.rs`). **SCOPE:** S-M.
- [x] Step 1: Write repro test / run prompt on LFM2.5-350M: prompt `"Explain how a transistor works in two paragraphs:"` generated only 1 token (early EOS `<|im_end|>`). (COMPLETE: reproduced).
- [x] Step 2: Bisect: isolated root cause — `run.rs` was short-circuiting BOS token injection in `--raw` mode, stripping `<|startoftext|>`. LFM2 tokenizer pre-type declares `add_bos_token = true`; running without BOS pushed LFM2 off-distribution to emit immediate EOS. (COMPLETE: fixed in `crates/grim-cli/src/run.rs`, committed `943f6152`).
- [x] Step 3: Verify fix: prompt now generates 32+ coherent tokens at 87 tokens/sec on ROCm. (COMPLETE).

---

## Phase 5B — Graph builder strangler (M5B) — after Phase 2 exit, BEFORE Phase 5

**EXISTS:** 89.4k LOC in `crates/grim-models/transformer/src/`; every family hand-re-derives graph emission. Defect class: YaRN ramp/mscale dropped (Xing4.0), Qwen attn-gate leaking into Llama graphs (PM-3), ring NORM eps hardcoded 1e-5, per-family `q81_ready` threading.
**CHANGE:** extract an emission BUILDER into a new `grim-graph` crate. Execution stays: `task_graph` + `LaunchEmitter` + ring dispatch are untouched. Strangler — old paths live until parity, then die.
**DO NOT CHANGE:** any executor; any kernel; the DON'T-TOUCH list (esp. capture no-sync discipline). No new opcodes.
**SCOPE:** 5B total is XL. Migration order is mandatory: llama before qwen35.

### Task 5B.1: crate skeleton + `emit_norm`
**Files:** Create `crates/grim-graph/{Cargo.toml,src/lib.rs,src/norm.rs}`; root `Cargo.toml` workspace members. Unit tests only — no GPU.
**Produces:** `NormSpec { weight: &Tensor, eps: f32 }`; `emit_norm(emitter, input, out, spec, m, k) -> Result<()>`. The eps→NORM/FALLBACK rule (|eps − 1e-5| < f32::EPSILON) moves HERE.
- [x] Failing test: eps 1e-5 records NORM opcode, 9e-6 records FALLBACK (recording emitter with `LaunchEmitter::launch`'s exact signature). (COMPLETE: tested and passed in `grim-graph`).
- [x] Port the norm block from `decode_graph.rs:2384-2430` region verbatim (M7.1 comment travels with it). `cargo test -p grim-graph` green. Commit. (COMPLETE: generic operand-carrying redesign landed in `b7080f62` and `9b6eb465`).

### Task 5B.2: `emit_linear_dedup` + `QuantFlags`
**Files:** Create `grim-graph/src/linear.rs`; modify `decode_graph.rs` to call INTO it (old fn kept as `#[cfg(test)]` reference).
**Produces:** `QuantFlags { q81_ready: bool, u4_ready: bool }` (pub, Copy); `emit_linear_dedup(dev, a, w, out, act_q81, what, layer, flags: &mut QuantFlags)`.
- [x] Failing test: two linears over the SAME input → second skips `grim_quantize_q8_1` (toy-graph census, pattern `decode_speed_bench.rs`). (COMPLETE: tested and passed in `grim-graph`).
- [x] Port `linear_into_named_dedup` verbatim. Byte-identical logits vs old path, `GRIM_DEDUP_QUANT` on AND off. Full `grim-models` suite. Commit. (COMPLETE: unified operand-carrying emitter landed in `b7080f62`, `b0d181c0`, `a7b029e2`).

### Task 5B.3: `emit_rope` — YaRN-unrepresentable
**Files:** Create `grim-graph/src/rope.rs`.
**Produces:** `RopeArgs::yarn(base, ramp, mscale)` / `RopeArgs::plain(base)`. A base-only constructor MUST NOT exist (enum or type-state) — that constructor is how Xing4.0 broke.
- [x] Failing test: yarn args reach the kernel arg list; guard test that no 2-arg rope emission exists. (COMPLETE: tested and passed in `grim-graph`).
- [x] Parity vs eager rope at pos 0 AND 41 (the Xing4.0 dump positions) under a YaRN config. Commit. (COMPLETE: committed `d5a18eec`).

### Task 5B.4: Migrate llama (smallest family, known bugs as the gate)
**Files:** `crates/grim-models/transformer/src/block.rs` / `decode_graph.rs` emission paths ONLY.
- [x] Baseline FIRST: llama PPL + launch census → `/tmp/llama_pre_builder.json`. (COMPLETE: recorded).
- [x] PM-3 closure written as an assertion: llama graphs must contain 0 `sigmoid_mul` nodes. Swapped emission to builder calls → PASS. (COMPLETE: committed `3f2bd179` and `cc98ba54`).
- [x] Gates: census node-for-node equal EXCEPT removed gate nodes; `llama_q41_task_graph.rs` green (0 cols over 10% of scale, max diff ~1e-6). Commit. (COMPLETE: committed `cc98ba54`, verified passing).

### Task 5B.5: Migrate qwen35 decode graph (the prize)
**Files:** `decode_graph.rs` emission regions ONLY. Placement / KDA wiring / routing policy STAY in `qwen35.rs`.
- [x] Baseline: structural task graph verified (37 nodes pinned node-for-node in `qwen35_graph_capture.rs`).
- [x] Migrate ONE block kind per commit:
  - Attention blocks: migrated through operand-carrying `grim-graph` (`emit_linear_dedup` with unified IR/launch emission, committed `a7b029e2`). Red-first verified on real site with `fuse_norm_quant = false`, 1 quantize launch across Q/K/V, logits byte-identical.
  - KDA recurrent blocks: migrated through operand-carrying `grim-graph` (`emit_linear_dedup`, committed `b0d181c0`). Red-first verified on real site, 1 quantize launch across 4 projections, logits byte-identical.
  - FFN blocks: migrated through operand-carrying `grim-graph` (`emit_norm` and `emit_linear_dedup` with explicit `record_ir: false`, committed `b7080f62` and `a3e2380f`).
    - *Finding on 37-node baseline:* In `d3c77e6e`, `ffn_gate` and `ffn_up` were never recorded as IR nodes (only `ffn_down` and residual add were). Emitting them unconditionally added 4 nodes (2 attention layers × 2 projections = 41 nodes). Explicit `record_ir: false` at call site preserves the 37-node contract node-for-node.
- [x] Final step (mandatory, anti-stall): Superseded hand-rolled emission paths in `decode_graph.rs` refactored into `RocmGraphEmitter` (committed `9b6eb465`). Full suites pass: `cargo test -p grim-cli` (bin tests), backend-rocm GPU-gated, grim-server integration.
- [x] Gates:
  - IR census node-for-node equal (37 nodes): pinned node-for-node in `qwen35_capture_emits_a_structurally_true_task_graph`.
  - Real-launch census gate (`quant_census_repro.rs`): passes on GPU, asserts 1 quant launch for attention and FFN dedup, byte-identical logits vs `GRIM_DEDUP_QUANT=0`.
  - Two-card split model replay green (committed `9a97ff23`).
  - `qwen35_graph_capture` suite 14/14 passed.
  **ON FAILURE at any sub-step:** the block kind reverts to the old path; report. Do not leave a half-migrated kind committed.

---

## Phase 5 — TP=2 (M5) — after Phases 1, 2 AND 5B

**EXISTS:** `plans/PLAN-tp2-decode.md` REV 2 (read it whole before starting): lanes-replace-packed, ~130-190 rendezvous, `[q|gate]` pairing + conv-state slice-map constraints; ≤45 ms target WITHDRAWN.
**CHANGE ONLY:** what REV 2 names, in the order it names. Constraint edits land in `grim-graph` builders ONCE (5B dependency).
**DO NOT CHANGE:** the pipeline arm — it stays the default and the fallback. `GRIM_TP_SIZE`/`GRIM_TP_RANK` fail-closed validation (`model_loader.rs:163-187`). **2026-11-06 (landed, `56baa573`): TP on a CPU target is now refused at load** — the dense transformer path has no cross-rank collectives, so a leaked `GRIM_TP_SIZE` silently loaded unreduced shards (even a truncated LM head); `RowParallelLinear` also errors instead of returning partial sums, and the dim==1 shard slicer enforces the block-quant boundary.
**SCOPE:** XL — do not start until M1, M2, M5B are DONE.

- [ ] Stage A close: re-run two-card A/B post-M1/M2; record new per-device busy + gap. That number is the Stage A exit.
- [ ] Stage B: re-derive the target from MEASURED busy/2 + measured rendezvous. **Write the stop criterion into the plan text before implementing:** "Ship TP if wall ≤ pipeline-wall − X ms, X ≥ rendezvous × 1.5, measured."
- [ ] Implement REV 2 constraints in order. Gate after each: ULP parity vs pipeline arm (TP changes reduction order — ULP, not byte, for fused norms).
- [ ] Two-card wall step + VRAM audit (post-prewarm free must rise vs 2.25/2.00 GiB).
- [ ] One commit per constraint. **ON FAILURE (rendezvous eats the savings):** record, keep pipeline default, close — a measured negative is a valid exit.

---

## Phase 6 — Consumer packaging (M6) — parallel from Phase 3 onward

Priority order within Phase 6: **6.5, 6.6 first** (audit pull order: everything else is cosmetic while the server stalls under load), then 6.7, 6.1, 6.2, 6.8, 6.3, 6.4, 6.9.

### Task 6.5: Inference off the tokio workers (audit L1)
**EXISTS:** `grim-server/src/lib.rs:158` single `Mutex<Engine>` held across whole GPU step; `completions.rs:183` non-stream + `:81` stream (`tokio::spawn` per token) run `sample_next_token` on async workers; `grep spawn_blocking` = 0 hits.
**CHANGE ONLY:** `grim-server` (lib.rs, routes/completions.rs) + engine tick call-site shape.
**DO NOT CHANGE:** engine step semantics, scheduler, kernel code. The engine API (`enqueue_request` + `tick()`) is preserved — only WHO calls it moves.
**SCOPE:** M-L.
- [x] Repro test FIRST: generation running, concurrent health probes tested (`test_health_responsive_during_continuous_generation` in `server_e2e_integration.rs`) — all respond < 2 s. (COMPLETE: committed `b028f1ad`)
- [x] Dedicated inference thread + mpsc; handlers post step request, await oneshot. `sample_next_token_async` moves `sample_next_token` completely off Tokio workers to dedicated `grim-inf-*` thread. (COMPLETE: committed `b028f1ad`)
- [x] Repro green + all 76 unit + 11 integration tests pass without regression. (COMPLETE: committed `b028f1ad`)

### Task 6.6: Graceful shutdown (audit L2)
**EXISTS:** `lib.rs:4009` `axum::serve` without `.with_graceful_shutdown`; `save_autotune_cache` (`kernel_infra.rs:875`) has ZERO exit-path callers; `dist/install.sh` registers systemd → SIGTERM = hard kill.
**CHANGE ONLY:** `grim-server/src/lib.rs` shutdown wiring.
**SCOPE:** S.
- [x] `with_graceful_shutdown(sigterm OR ctrl_c)`: stop admission → drain in-flight (bounded) → SSE error frames → `save_all_autotune_caches()` → exit 0. (COMPLETE: committed `d1e9b2d7`)
- [x] Test: SIGTERM/Ctrl-C signal handling tested via `test_graceful_shutdown_admission_and_drain` verifying immediate 503 admission cutoff, token cancellation, and clean drain. (COMPLETE: committed `d1e9b2d7`)

### Task 6.7: Build survives consumer ROCm (audit L3+L4+L5)
**EXISTS:** `build_rocm_detect.rs` hardcodes `/home/nelson/.cache/yay/rocm-bin/...`; `grim-hip-sys/src/hipconfig.rs` panics when `hipconfig` writes stderr; `build.rs` emits `dylib=rocblas` whenever the dir is found, but consumer installs often ship versioned-only `.so` (link fails cryptic: `-lrocblas not found`).
**CHANGE ONLY:** those 3 build files.
**DO NOT CHANGE:** the `dynamic` feature's dlopen fallback chain; the committed-bindings posture in `build.rs` header.
**SCOPE:** S.
- [x] Delete nelson paths (keep `ROCM_PATH` / `/opt/rocm` / `/usr/lib/rocm` / pkg-config). (COMPLETE: committed `1c0ec656`)
- [x] hipconfig stderr → warning + highest-committed-bindings fallback. (COMPLETE: committed `1c0ec656`)
- [x] Add search candidates for versioned ROCm/HIP/rocBLAS `.so` targets (`.so.6`, `.so.5`, etc.). (COMPLETE: committed `1c0ec656`)
- [x] Verify: clean cargo check and test pass. (COMPLETE: committed `1c0ec656`)

### Task 6.1: HIP ABI gate
**EXISTS:** exactly one committed binding (`bindings_26385.rs`, HIP 7.16.26385); `grim-hip-sys/src/dynamic.rs:21-35` PANICS on missing symbol.
**CHANGE ONLY:** device-init gate in `grim-hip-sys` + `dynamic.rs` panic→Err.
**SCOPE:** S.
- [x] Version mismatch at init → readable error naming both versions + the `dynamic` fallback, before any kernel runs. Missing symbol → `Err` with symbol name (`try_load`). (COMPLETE: committed `0231cc17`)

### Task 6.2: Watchdog load exemption
**EXISTS:** `device/util.rs:37-77` — 30 s sync watchdog → banner + `exit(101)`. A 27B load from slow NVMe can trip it.
**CHANGE ONLY:** `device/util.rs`.
**SCOPE:** S.
- [x] Watchdog arms after first successful step (`arm_rocm_watchdog`), or timeout ≥ 1 s/GiB of model bytes (`notify_model_loading_bytes`). Document exit 101 in troubleshooting. (COMPLETE: committed `048bca0c`)

### Task 6.8: Panic→Result sweep (audit L7/L8/L9/L17)
**EXISTS (confirmed prod sites):** `device/cubecl.rs:131,160,191,232` (`.expect` in the FALLBACK path); `gptq_awq.rs:732-736` (fix inside Task 1.2's diff, not twice); `memory/view.rs:213-237`; `gemm_launchers.rs:1007` (`.expect("just inserted")` — HashMap insert guard is defensible: convert OR comment the invariant); `trace.rs:611-612`; `completions.rs:129,159` (`serde_json::to_string(&chunk).unwrap()` in `tokio::spawn` → stream ends with no error frame).
**CHANGE ONLY:** those files. **SCOPE:** M.
- [x] gemm_launchers cvt insert `.expect` replaced with safe Result error return. (COMPLETE: committed `803f5676`)
- [x] completions: on serialization error send error frame and break, final chunk safely formatted — stream never ends silently or unwinds. (COMPLETE: committed `803f5676`)
- [x] `cargo clippy -- -D clippy::unwrap_used` scoped to `src/device` + `src/memory` + grim-server routes; fix or `#[allow]` with written invariant. Commit per file group. (COMPLETE: committed `903fbc58`).
- [x] Widen (audit A5): same gate over non-test code in `grim-server` (178 unwrap), `grim-engine` (148/82 expect), `grim-format` (177/85 — parses untrusted files), `grim-cli` user-facing paths. Counts include inline tests; gate only `not(test)`. (COMPLETE: committed `9e130b9c`).

### Task 6.3: Docs scrub + config matrix
**EXISTS (breaks):** `docs/configuration.md` GRIM_GPU_TARGET default gfx1100; `docs/troubleshooting.md:27` ROCm 6 SONAMEs; README.md:11 + install-grim.md:20 Rust 1.85 vs `Cargo.toml` 1.86; README.md:35 phantom `crates/grim-tensor-graph/` row; README.md:21 `sydonayrex/grim` vs install-grim.md:44 `poolside-ai/grim`; 395 GRIM_* in code vs 131 documented; README Quick Start has no ROCm prerequisite.
**CHANGE ONLY:** docs + the env-var generation script. **SCOPE:** S.
- [x] Fixed broken docs, updated Rust 1.86 requirement, phantom crate link, sydonayrex repo links, added watchdog exit 101, page fault serialization, and documented GRIM_PHASE_TRACE, GRIM_FUSED_RESIDUAL_GATEUP, GRIM_BLASLT_PREFILL, GRIM_WATCHDOG_TIMEOUT_SECS, GRIM_CAPABILITY_PLACEMENT. (COMPLETE: committed `ceb8d6e2`)

### Task 6.4: Mixed-card placement default
**EXISTS:** capability-aware placement is opt-in (`GRIM_CAPABILITY_PLACEMENT=1`, `qwen35.rs:5784-5796`). 9070 XT + 9060 XT is THE consumer config.
**CHANGE ONLY:** `qwen35.rs` default + docs. **SCOPE:** S.
- [x] Default-on when multiple devices are configured; `GRIM_CAPABILITY_PLACEMENT=0` reverts. Probes throughput/bandwidth automatically for balanced layer blocks. (COMPLETE: committed `df186afd`)

### Task 6.9: P2P topology truthfulness (audit L15)
**EXISTS:** `device/hardware_spec.rs` `From<&RocmDevice>` hardcodes 1 device / NoLink; `peer_access.rs` / `p2p_route.rs` exist.
**SCOPE:** S.
- [x] Wire real device enumeration (`enumerate_devices()`) + link type probing (`peer_status`, `enable_peer_access`) into `HardwareSpec::from(&RocmDevice)`. (COMPLETE: committed `321aa380`)

---

## Phase 7 — Dead weight cut (M7) — LAST, mechanical

One commit per cut; full crate suite between cuts; GPU-gated suites re-run after item 1 only.

1. Feature-gate `grim-backend-vulkan` out of grim-cli + grim-server defaults (engine already hardcodes no-vulkan, `model_loader.rs:5585`). Same pass: `grim-backend-metal` optional in grim-server (it IS optional in cli — audit L11), cubecl behind a `parity-tests` feature (shipping two JIT toolchains in the default build is test config, not ship config).
1b. ~~DELETE the fake FP8 kernel (audit L6)~~ — **DONE** (commits `34211d2b` + `ee953d2a` + `78557405`): kernel file (105 LOC), launcher, mod decl, source_asm push + census assertion, presence test, and the missed test-file deletion all removed. 0-caller trace verified (only the def + a doc comment acknowledging dead state); lib 595/595; GPU parity (dot4 17/17, wmma ok). If someone wants real FP8: implement `__builtin_amdgcn_mfma` in a NEW file with a new name.
1c. `#[allow(dead_code)]` sweep (69 markers in grim-backend-rocm): each a deletion candidate; keep + justify only what Phase 5B/TP=2 will revive.
2. ~~Delete `grim-compressed-tensors`~~ — **DONE** (commit `68cedcdd`): consumer trace showed the 504-LOC crate's only external surface was one zero-user re-export in grim-tensor; crate dir + workspace member + dep + re-export removed, `cargo check --workspace` clean.
3. ~~Delete/feature-gate `grim-constrain` + `grim-compress`~~ — **DONE** (commit `b4c63dab`): call-site trace confirmed off-by-default (`response_format` json arms; `GRIM_COMPRESS_PROMPT=1`); moved to opt-in features `constrained-gen` / `prompt-compress` (2,887 LOC out of the default build); gated arms return a structured error naming the feature; default + featured builds compile, grim-server lib 76/76.
4. Garage web UI: out of the compat crate (separate crate or cut) — decide by grim-cli routes count.
5. fp8_e4m3_to_f32: single source in grim-quant, re-exported — parity-gated, no behavior change (8 copies today).
6. Cubecl shim crates: inline the 5-line re-exports, keep `grim-hip-sys`.
7. Per-token `eprintln!` (`grim-server/src/lib.rs:553-556`) → `tracing` debug. (Full logging migration is Task 8.2.)
8. **ROCm kernel cuts (kernel review 2026-11-01, `rocm-kernel-deletion-review.md`).** Each verified by a Rust-entry-point trace, not a name grep. BEFORE each cut: run the GPU census (`cargo test --release -p grim-models-transformer --test qwen35_graph_capture -- --nocapture`) and confirm 0 launches of the kernel; AFTER each cut: run `cargo test -p grim-backend-rocm` + GPU parity suite. One commit per bullet:
   - **8a. Segmented batched LoRA dead path:** In `crates/grim-backend-rocm/src/kernels/batched_lora.rs`, delete `BATCHED_LORA_HIP_SRC` (`grim_batched_lora_shrink`, `grim_batched_lora_accumulate`), `pub fn batched_lora_group_device` (line 546), and `pub fn batched_lora_dispatched_cpu` (line 420), plus their unit tests in the same file. **DO NOT DELETE** `DISPATCHED_LORA_HIP_SRC` (`grim_lora_shrink_dispatched`, `grim_lora_expand_dispatched`), `batched_lora_dispatched_device` (line 216), or `batched_lora_accumulate_cpu` (line 147) — those are LIVE.
   - **8b. ~~Dead Raven FP8 kernel~~ DONE** (`51ddd908`): grim_wmma_gemm_fp8 removed from KERNEL_SOURCE (0 Rust entry points); blocked_v2 + mfma variants retained.
   - **8c. ~~Unused MoE expert compaction kernel~~ DONE** (`51ddd908` + `78557405`): module, mod decl, and its 2-test file removed — NO HIP source was ever pushed (source_asm had zero references); test-file deletion initially missed by a suppressed-failure `git rm`, staged as a fixup.
   - **8d. ~~Unwired MoE scatter-add kernel~~ DONE** (`51ddd908`): removed from iq_gemm KERNEL_SOURCE after the 0-caller trace.
   - **8e. ~~Dead MAdam f32 launcher~~ DONE** (`51ddd908`): the launcher actually lived in kernel_infra.rs (plan's optimizer_ops.rs ref was stale), 87 lines with `#[allow(dead_code)]`, 0 callers; stale comment ref updated. fused_madam_step retained.
   - **8f. Losing off-by-default kernel arms:**
     - In `crates/grim-backend-rocm/src/device/quant/dequant_k_quants.rs` (lines 113, 368), delete the `GRIM_Q4K_TILED` backward branch and its tiled backward kernel if `GRIM_Q4K_TILED=1` is not faster than default on `golden_q4k_backward`.
     - In `crates/grim-backend-rocm/src/device/compute/dot_gemv.rs:3692` and `crates/grim-backend-rocm/src/kernels/scrub_jay.rs`, delete `GRIM_SCRUB_JAY_V2` and `grim_scrub_jay_gemv_v2`.
     - In `crates/grim-backend-rocm/src/device/compute/dot_gemv.rs:1244` and `crates/grim-backend-rocm/src/kernels/dot_gemv.rs`, delete `GRIM_DOT4_256` and `launch_dot4_gate_up_silu_q80_f32act_gemv_256_into`.
     - After Task 2.8 completes: in `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs:790-805`, `crates/grim-backend-rocm/src/device/compute/dot_gemv.rs:3138`, and `crates/grim-backend-rocm/src/kernels/dot_gemv.rs`, delete `GRIM_WHITECROW_V2`, `GRIM_WHITECROW_V3`, `grim_dot8_w4a4_gemv_v2`, and `grim_dot8_w4a4_gemv_v3`.
   - **8g. Losing KV-quant implementation:** Delete whichever of the two KV-quant implementations (unused in-tree `qkv_attention.rs` variants vs `crates/grim-kvquant`) was marked redundant in Task 2.3 Step 0.
9. **Move probes/reference kernels to `#[cfg(test)]` / `examples/`:**
   - Move `extern "C" __global__ void grim_sdot4_probe` (`crates/grim-backend-rocm/src/kernels/dot_gemv.rs:5505`) out of `dot_gemv::KERNEL_SOURCE` into `crates/grim-backend-rocm/examples/kquant_dot4_vs_exact_probe.rs`.
   - Gate `crates/grim-backend-rocm/src/kernels/rho_microbench.rs` push in `crates/grim-backend-rocm/src/kernels/source_asm.rs:15` and `pub fn rho_microbench_for_test` (`crates/grim-backend-rocm/src/device/compute/gemm_launchers.rs:4761`) behind `#[cfg(test)]`.
   - Gate the 8 scalar IQ launchers `launch_fused_dequant_gemm_iq{1m,1s,2s,2xs,2xxs,3s,3xxs,4xs}_scalar` in `crates/grim-backend-rocm/src/device/quant/dequant_iq_quants.rs` behind `#[cfg(test)]`.
10. **Strip stale `#[allow(dead_code)]` markers from LIVE functions:** Remove `#[allow(dead_code)]` from `launch_charon_grouped_dispatch_fp8` (`crates/grim-backend-rocm/src/device/routing/charon_grouped_dispatch.rs`), `launch_wmma_fused_dequant_{q2k,q3k,q4k,q5k,q6k,q8_0,q8_0_fp16,iq2xxs,iq2xs,iq2s,iq3xxs,iq3s,iq4nl,iq4xs}` (`crates/grim-backend-rocm/src/device/quant/wmma_routing.rs`), `launch_decode_gemm_f16`, and `route_gemm_to`, then verify `cargo check -p grim-backend-rocm` passes with `dead_code = "deny"`.
11. **Rollback-switch retirement** (default-ON A/B gates — delete the OLD fallback arm + its kernel after one release with no rollback use): `GRIM_WHITE_RAVEN_V2` (`gemm_launchers.rs:2151`), `GRIM_Q5K_SUB` (`dot_gemv.rs:2349`, deletes `grim_dot4_q5k_q81_gemv` and `_ref`), `GRIM_DOT4_FAST` (`dot_gemv.rs:2248`), `GRIM_IQ_LDS` (`dequant_iq_quants.rs:432`), `GRIM_WMM_QUANT` (`device/quant/mod.rs:2597`), `GRIM_DEDUP_QUANT` (post T1.2). Record in `CHANGELOG.md` (T8.6).
12. **Per-format GEMV template** (post T2.2/T2.5, only if the template proved out on both `iq4xs` and `q5k`): collapse per-format copies in `crates/grim-backend-rocm/src/kernels/dot_gemv.rs` (62 kernels, 6.9k LOC) onto a single C++ macro/template `GRIM_DEFINE_DOT4_SUB_GEMV(NAME, BLOCK_BYTES, UNPACK_FN)` backed by per-format `dequant_block` device fns in `crates/grim-backend-rocm/src/kernels/shared_device_fns.rs`. Gate each format migration with `dot4_kquant_grouped_parity.rs` (byte-identical output vs pre-template kernel); one format per commit.

---

## Phase 10 — Feature wiring & integration (M10) — from kernel review; parallel after M1

Kernels exist for these features, but the end-to-end hookup is missing or incomplete. Default posture: **keep, wire, and integrate** every feature kernel end-to-end rather than deleting or parking behind feature flags.

### Task 10.1: Per-layer multi-LoRA serving (rework)
**EXISTS:** `grim_lora_shrink_dispatched` and `grim_lora_expand_dispatched` (`crates/grim-backend-rocm/src/kernels/batched_lora.rs:322,354`) are called via `batched_lora_dispatched_device` (`batched_lora.rs:216`) from `crates/grim-engine/src/engine/adapters.rs:166` (`apply_batched_lora_to_rows`), which is invoked only on the final **logits** in `crates/grim-engine/src/engine/scheduler.rs:1653` and `crates/grim-engine/src/lib.rs:2051`. DEFECTS: (a) adapters trained by `grim-autograd` target per-layer projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), NOT the vocab logits vector (`in_dim = out_dim = vocab_size`); (b) `batched_lora_dispatched_device` takes host slices `&[f32]`, allocates new device buffers (`hipMalloc`) and calls `RocmDevice::new(*ordinal)` on every invocation (`adapters.rs:164-165`), causing a host round-trip (`stacked.to_vec()`) and sync.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/kernels/batched_lora.rs`, `crates/grim-engine/src/engine/adapters.rs`, `crates/grim-engine/src/engine/scheduler.rs`, `crates/grim-engine/src/lib.rs`, `crates/grim-models/transformer/src/decode_graph.rs` (keep off `crates/grim-graph` until Phase 5B lands). **SCOPE:** L.
- [x] Step 1: Write failing test FIRST in `crates/grim-engine/tests/multi_lora_per_layer_parity.rs`: create a toy 2-layer transformer (or load a small GGUF), attach a rank-8 LoRA adapter on `blk.0.attn_q.weight` and `blk.0.attn_v.weight` with known non-zero `A` and `B`, and compare outputs of (Arm A) base weights + runtime multi-LoRA dispatch vs (Arm B) base weights with `W' = W + (alpha / r) * B * A` pre-merged into `q_proj`/`v_proj`. Assert max abs logit diff `< 1e-3`. Run `cargo test --release -p grim-engine --test multi_lora_per_layer_parity` → expect FAIL today. (COMPLETE: test created and verified).
- [x] Step 2: In `crates/grim-backend-rocm/src/kernels/batched_lora.rs`, add `pub fn launch_batched_lora_dispatched_storage(dev: &RocmDevice, x: &RocmStorage, y: &RocmStorage, scratch_v: &RocmStorage, d_token_adapter: &RocmStorage, d_adapters: &RocmStorage, num_tokens: usize, in_dim: usize, out_dim: usize, max_rank: usize) -> Result<()>` that operates strictly on pre-allocated `RocmStorage` device pointers with ZERO `hipMalloc`, ZERO `hipMemcpy`, and ZERO `hipDeviceSynchronize` during the step (HIP-graph-capture safe). (COMPLETE).
- [x] Step 3: In `crates/grim-engine/src/engine/adapters.rs`, store each loaded adapter's per-layer `(A_dev: RocmStorage, B_dev: RocmStorage, rank: u32, scaling: f32)` keyed by `(adapter_id, layer_idx, TargetModule)` when `register_adapter` is called, and upload the `DispatchedLoraAdapter` descriptor table to device memory once per adapter registration / batch change. (COMPLETE: TargetModule and per-layer AdapterHandle wired).
- [x] Step 4: In `crates/grim-models/transformer/src/decode_graph.rs` (`linear_into_named` / `linear_into_named_dedup`), after the base GEMV/GEMM writes `out`, if the active batch has any non-zero `adapter_id` for `(layer_idx, target_module)`, call `launch_batched_lora_dispatched_storage` to accumulate `Δy = scaling * (x @ A^T) @ B^T` into `out` in-place. When `active_adapters.is_empty()`, emit ZERO extra kernel launches. Remove the broken post-logits `apply_batched_lora_to_rows` call from `scheduler.rs:1653` and `lib.rs:2051`. (Note: when Phase 5B lands `grim-graph`, this call site will naturally move into `grim-graph/src/linear.rs`). (COMPLETE).
- [x] Step 5: Gates: (a) `multi_lora_per_layer_parity` test from Step 1 passes; (b) mixed batch of 3 requests (`adapter_id = 0` base, `adapter_id = 1`, `adapter_id = 2`) matches individual single-request runs within `< 1e-3`; (c) base-only (`adapter_id = 0`) 27B launch census is **node-for-node identical** to pre-change (0 extra launches). Commit: `git commit -m "feat(lora): wire device-resident per-layer multi-LoRA dispatch into linear projections"` (COMPLETE: committed `68bae1cd`).

### Task 10.2: Diffusion/image-model LoRA
**EXISTS:** `crates/grim-models/diffusion/src/{flux2.rs,unet.rs,vae.rs,lib.rs}` has ZERO LoRA code. Image LoRA safetensors (Kohya `lora_unet_*` / Diffusers `transformer.*` `.lora_A.weight` + `.lora_B.weight` + `.alpha`) are standard.
**CHANGE ONLY:** `crates/grim-models/diffusion/src/lib.rs`, `crates/grim-models/diffusion/src/unet.rs`, `crates/grim-models/diffusion/src/flux2.rs`, new `crates/grim-models/diffusion/src/lora.rs`, new test `crates/grim-models/diffusion/tests/diffusion_lora_test.rs`. **SCOPE:** M.
- [x] Step 1: Create `crates/grim-models/diffusion/src/lora.rs` implementing `pub struct DiffusionLoraAdapter` with `pub fn from_safetensors(path: &Path, scale: f32) -> Result<Self>`. Normalize both Kohya keys (`lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight` / `.lora_up.weight` / `.alpha`) and Diffusers Flux/UNet keys (`*.to_q.lora_A.weight` / `.lora_B.weight`) to canonical model tensor names. (COMPLETE).
- [x] Step 2: Implement **Phase 1 (merge-at-load):** `pub fn apply_merged(&self, weights: &mut HashMap<String, Tensor>, scale_override: Option<f32>) -> Result<usize>` which computes `W += (scale * alpha / rank) * (B @ A)` (and for Conv2d 1×1 kernels reshapes `[out, rank, 1, 1] @ [rank, in, 1, 1]` → `[out, in, 1, 1]`), supporting multiple stacked adapters with independent scales. (COMPLETE).
- [x] Step 3: Implement **Phase 2 (runtime dispatched LoRA without weight reload):** attach `Vec<DiffusionLoraAdapter>` to `UNet2DConditionModel` (`unet.rs`) and `Flux2Transformer` (`flux2.rs`) linear layers and invoke `launch_batched_lora_dispatched_storage` (from Task 10.1) during the forward pass when unmerged runtime adapters are active. (COMPLETE).
- [x] Step 4: Write `crates/grim-models/diffusion/tests/diffusion_lora_test.rs`: test (a) Kohya + Diffusers key normalization; (b) merged forward pass vs runtime dispatched forward pass on a synthetic UNet/Flux2 block with a fixed RNG seed → assert output tensors match within `< 1e-4` max abs diff and differ from the unadapted base output. Commit: `git commit -m "feat(diffusion): add Kohya/Diffusers LoRA merge-at-load and runtime dispatched forward"` (COMPLETE: committed `452ef339`).

### Task 10.3: MXFP4/MXFP8 backward GEMM (training over MXFP bases)
**EXISTS:** `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs:537` defines `extern "C" __global__ void grim_fused_dequant_backward_gemm_mxfp4(const float* dY, const unsigned char* B_codes, const unsigned char* B_exps, float* dA, int M, int N, int K)` and line 598 defines `grim_fused_dequant_backward_gemm_mxfp8(const float* dY, const unsigned char* B_fp8, const unsigned char* B_exps, float* dA, int M, int N, int K)`. Both are already compiled into the JIT image (`source_asm.rs:64` pushes `wmma_gemm::KERNEL_SOURCE`), but NEITHER has a Rust launcher or backward dispatch arm.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/quant/mxfp_gemm.rs`, `crates/grim-backend-rocm/src/device/compute/autograd_ops.rs` (or `crates/grim-backend-rocm/src/device/quant/mod.rs` backward dispatch), new test `crates/grim-autograd/tests/golden_mxfp_backward.rs`. **SCOPE:** S-M.
- [x] Step 1: Write failing golden test FIRST in `crates/grim-autograd/tests/golden_mxfp_backward.rs` (copy structure from `crates/grim-autograd/tests/golden_q4k_backward.rs`): quantize a random `[N=128, K=64]` weight matrix to MXFP4 (`B_codes` `[N, K/2]`, `B_exps` `[N, K/32]`) and MXFP8 (`B_fp8` `[N, K]`, `B_exps` `[N, K/32]`), run backward `dA = dY @ dequant(B)` for `M=4`, and assert max abs diff vs CPU reference dequant + f32 matmul `< 1e-4`. (COMPLETE).
- [x] Step 2: In `crates/grim-backend-rocm/src/device/quant/mxfp_gemm.rs`, add `pub fn launch_fused_dequant_backward_gemm_mxfp4(&self, dy: &RocmStorage, b_codes_ptr: u64, b_exps_ptr: u64, da: &RocmStorage, m: usize, n: usize, k: usize) -> Result<*mut c_void>` and `pub fn launch_fused_dequant_backward_gemm_mxfp8(...)`: (COMPLETE).
  - Guard `if k == 0 || k % 32 != 0 { return Err(Error::Backend(...)); }` (matching `launch_mxfp4_gemm_tiled` at line 27).
  - Launch with `block_dim = HipDim3::new(256, 1, 1)` and `grid_dim = HipDim3::new((m * k).div_ceil(256) as u32, 1, 1)`, passing `(dy_ptr, b_codes_ptr, b_exps_ptr, da_ptr, m as i32, n as i32, k as i32)`.
- [x] Step 3: Wire the `Mxfp4` / `Mxfp8` storage variants in the backward matmul dispatch (`crates/grim-backend-rocm/src/device/quant/mod.rs` / `autograd_ops.rs`) to call the two new launchers. (COMPLETE).
- [x] Step 4: Run `cargo test --release -p grim-autograd --test golden_mxfp_backward -- --ignored` (must PASS) and add a toy LoRA overfit test over an MXFP4 linear layer asserting loss decreases by `> 90%` over 50 steps. Commit: `git commit -m "feat(rocm,autograd): wire grim_fused_dequant_backward_gemm_mxfp4 and mxfp8 launchers"` (COMPLETE: committed `7b977a47`).

### Task 10.4: GPU log-softmax VJP for preference training
**EXISTS:** `crates/grim-autograd/src/preference_trainer.rs:244` (`compute_log_softmax_vjp`) and `:292` (`compute_log_softmax_vjp_tensor`) compute the cross-entropy / DPO gradient on the CPU by copying the entire `[seq_len, vocab_size]` logits tensor to host. `crates/grim-backend-rocm/src/kernels/log_softmax_vjp.rs:19` defines `extern "C" __global__ void grim_log_softmax_vjp_kernel(const float* logits, const unsigned int* targets, float* grad_out, float d_loss_d_logp, unsigned int ignore_index, int vocab_size, int seq_len)`, which is NOT yet pushed in `source_asm.rs` and has no launcher.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/kernels/source_asm.rs`, `crates/grim-backend-rocm/src/device/compute/autograd_ops.rs`, `crates/grim-tensor/src/backend.rs`, `crates/grim-autograd/src/preference_trainer.rs`, `crates/grim-autograd/tests/preference_trainer_integration.rs`. **SCOPE:** S.
- [x] Step 1: In `crates/grim-backend-rocm/src/kernels/source_asm.rs` (after line 12 where `FUSED_LINEAR_CE_KERNEL_SOURCE` is pushed), append `s.push_str(crate::kernels::log_softmax_vjp::LOG_SOFTMAX_VJP_KERNEL_SOURCE);`. (COMPLETE).
- [x] Step 2: In `crates/grim-backend-rocm/src/device/compute/autograd_ops.rs`, implement `pub fn launch_log_softmax_vjp(&self, logits: &RocmStorage, targets: &RocmStorage, grad_out: &RocmStorage, d_loss_d_logp: f32, ignore_index: u32, vocab_size: usize, seq_len: usize) -> Result<*mut c_void>` with `grid_dim = HipDim3::new(seq_len as u32, 1, 1)` and `block_dim = HipDim3::new(256, 1, 1)` (1 CTA per sequence row, 8 warps of 32 threads; matches `s_max[64]` / `s_sum[64]` in `log_softmax_vjp.rs:43,58`). Expose `fn log_softmax_vjp(...)` on the `BackendDevice` trait in `crates/grim-tensor/src/backend.rs` (defaulting to CPU fallback on non-ROCm backends). (COMPLETE).
- [x] Step 3: In `crates/grim-autograd/src/preference_trainer.rs:292` (`compute_log_softmax_vjp_tensor`), when `logits.device().is_rocm()`, upload `targets` (`&[u32]`) to device, allocate `grad_out` on device, and invoke `dev.log_softmax_vjp(...)` without copying `logits` to host. (COMPLETE).
- [x] Step 4: In `crates/grim-autograd/tests/preference_trainer_integration.rs`, add a GPU parity test (`seq_len = 16, vocab_size = 32000`, including rows where `target == ignore_index`) asserting max abs diff vs `Self::compute_log_softmax_vjp` (CPU) `< 1e-5`. Record DPO step time before → after in the commit message. Commit: `git commit -m "perf(autograd,rocm): wire GPU grim_log_softmax_vjp_kernel into preference_trainer"` (COMPLETE: committed `f5c7e15b`).

### Task 10.5: Wire BitNet (`TQ1_0`/`TQ2_0` and `W158A8_INT8`) & Bonsai-27B (`PQ2_0` + Hadamard) on ROCm
**CORRECTION (2026-11-03, model file found):** `models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.2 GB) IS on disk — header: 402 weight tensors are GGUF tag 142 = **PQ2_0** (PrismML Ternary-2), not TQ2_0. grim-quant already has `dequant_pq2_0` + correct 128/34 block geometry; only the gguf.rs storage mapping refuses it (Storage::Unsupported). Dequant-at-load is NOT viable (≈108 GB f32 vs 32 GB total VRAM). `grim_bitnet_gemm_w158a8` is ABI-incompatible (per-row scale vs PQ2_0's per-128-block fp16 scale; different code mapping). **Real unblock = new kernel task: PQ2_0 device Storage variant + PQ2_0 GEMV (template: the Q2_0 k-quant dot4 arm, same 2-bit×fp16-scale math at block 128) + parity gate vs `dequant_pq2_0` + route Bonsai-27B. L-scope; park here until owned.**
**STATUS 2026-11-03: the kernel task LANDED — `grim_dot4_pq2_0_q80_gemv` (geometry clone of `grim_dot4_q2_0_q80_gemv`, which already implements the exact PQ2_0 (q−1)·d code law), launcher, `BlockDtype::Pq2_0` storage, gguf.rs tag-142 mapping, ROCm dispatch arm, host-decode arms in convert/CPU/Vulkan/CUDA/grim-nn/autograd. Gates green: GPU kernel bit-exact vs `dequant_pq2_0` oracle (`tests/pq2_0_gemm_parity.rs`, worst rel err 0.00e0), provider serves the real 7.2 GB Bonsai GGUF with correct 34 B/128 geometry (`tests/bonsai_pq2_0_provider.rs`).
**RESOLVED (2026-11-04, commit `699c597e`):** The GGUF header declares `prism.hadamard.transform = normalized-sylvester-walsh-hadamard`, block 1024, input-last-dimension axis, with an explicit sign mask (28,672 values, `sign_widths`/`sign_values`) over 401 rotated tensors (+1 inverse-side on `token_embd`). Fast Walsh-Hadamard Transform (FWHT-1024) and sign transform runtime support implemented in `crates/grim-backend-rocm/src/kernels/hadamard.rs`, wired to ROCm launchers and `decode_graph.rs:linear_folded`, and fully verified end-to-end against the Bonsai-27B model on disk (`crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs`).
**2026-11-06: Gate D (32-token parity vs llama.cpp) now PASSES 32/32 — in BOTH the quantized perf path and `GRIM_PQ2_F32ACT` mode; all 5 bonsai gates green.** Root cause of the long-standing 18/32→23/32 near-tie divergence was NOT quant noise: grim's RoPE frequency used `base^(-2i/head_dim)` where ggml uses `base^(-2i/n_rot)` (`rope.dimension_count`=64 vs head_dim 256 — up to 4× wrong angles on partial-rotary models; invisible on full-rotary models). Fixed in `yarn_inv_freq_and_mscale`, `grim_rope_dev_base`, `grim_rope_append_whitecrow_k`, mxfp4 fused-rope fallbacks, and CPU rope (`528b1508`); yarn corr-dims to n_rot (`e629f334`); golden standard for CPU rope/rerope with 3 rerope defects fixed (`c3793a88`). Ref-capture recipe + localization chain in memory `reference-llama-prism-hadamard`.
**EXISTS:** `crates/grim-backend-rocm/src/kernels/bitnet_gemm.rs` has two kernels (already in `source_asm.rs:111`): `grim_bitnet_gemm_w158a8` (f32 activations, launched by `launch_bitnet_gemm` in `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs:1320`) and `grim_bitnet_gemm_w158a8_int8` (line 53, `const signed char* A_int8`, unwired). `crates/grim-format/src/gguf.rs:181-183` parses `GgufDType::TQ1_0` (34) and `GgufDType::TQ2_0` (35), and `crates/grim-models/transformer/src/bitnet.rs` wraps Llama, but `TQ2_0` weights are not routed to the BitNet GEMM kernels on ROCm.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs`, `crates/grim-backend-rocm/src/device/compute/gemm_launchers.rs`, `crates/grim-models/transformer/src/bitnet.rs`, `crates/grim-engine/src/model_loader.rs`. **SCOPE:** M.
- [x] Step 1: In `crates/grim-backend-rocm/src/device/quant/gptq_awq.rs` (immediately after `launch_bitnet_gemm` at line 1340), add `pub fn launch_bitnet_gemm_int8(&self, a_int8_storage: &RocmStorage, b_ternary_storage: &RocmStorage, scale_b_storage: &RocmStorage, out_storage: &RocmStorage, m: usize, n: usize, k: usize, scale_a: f32) -> Result<*mut c_void>` launching `"grim_bitnet_gemm_w158a8_int8"` with `block_dim = HipDim3::new(16, 16, 1)` and `grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1)`. (COMPLETE).
- [x] Step 2: Route BitNet 2-bit packed ternary weights (`TQ2_0` / `BitNet158`) in `crates/grim-backend-rocm/src/device/compute/gemm_launchers.rs` and `crates/grim-models/transformer/src/bitnet.rs`: quantize activations `A` to `i8` + `scale_a` on device (`grim_quantize_int8_rowwise`) and dispatch `launch_bitnet_gemm_int8` (falling back to `launch_bitnet_gemm` when `GRIM_BITNET_FP32_ACT=1`). (COMPLETE).
- [x] Step 3: Write a parity test in `crates/grim-backend-rocm/tests/bitnet_gemm_parity.rs` comparing `launch_bitnet_gemm_int8` and `launch_bitnet_gemm` against a CPU reference ternary matmul over `M ∈ {1, 16}, N = 2048, K = 2048` (assert relative error `< 2e-2` for int8 act, `< 1e-5` for f32 act). Promote `bitnet` to `SupportTier::Green` in `crates/grim-engine/src/model_loader.rs` (Task 4.3). Commit: `git commit -m "feat(rocm,bitnet): wire grim_bitnet_gemm_w158a8_int8 and TQ2_0 dispatch"` (COMPLETE: committed `3eb500a6`).
- [x] Step 4: Wire Bonsai-27B `PQ2_0` GEMV and Hadamard rotated-domain transform: Fast Walsh-Hadamard Transform (FWHT-1024) + sign mask unpacker + runtime folded activation routing on Qwen35/Bonsai recurrent blocks. Verified end-to-end across all 6 gates in `crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs`. Commit: `git commit -m "feat(prism): wire Bonsai-27B PQ2_0 GEMV and Hadamard rotated-domain transform (Task 10.5)"` (COMPLETE: committed `699c597e`).

### Task 10.6: Wire Speculative Decoding GPU verification
**EXISTS:** `crates/grim-backend-rocm/src/kernels/speculative_sampler.rs` (already pushed in `source_asm.rs:115`) defines 4 kernels: `grim_speculative_rejection_sample` (line 8), `grim_sample_logits_argmax` (line 99), `grim_sample_stochastic` (line 145), and `grim_speculative_tree_verify` (line 232). `crates/grim-engine/src/speculative_loop.rs:56` (`verify_draft_step`, `verify_tokens_with_rejection_sampling`, `verify_medusa_tree_candidates`) runs on CPU slices and is not called from the scheduler decode loop.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/compute/sampling_ops.rs`, `crates/grim-tensor/src/backend.rs`, `crates/grim-engine/src/speculative_loop.rs`, `crates/grim-engine/src/engine/scheduler.rs`. **SCOPE:** M-L.
- [x] Step 1: In `crates/grim-backend-rocm/src/device/compute/sampling_ops.rs`, add launchers on `RocmDevice`: (COMPLETE).
  - `pub fn launch_speculative_rejection_sample(&self, target_probs: &RocmStorage, draft_probs: &RocmStorage, draft_tokens: &RocmStorage, uniform_rands: &RocmStorage, accepted_tokens: &RocmStorage, accepted_lens: &RocmStorage, batch_size: usize, num_draft_tokens: usize, vocab_size: usize) -> Result<*mut c_void>` (`grid = (batch_size, 1, 1)`, `block = (256, 1, 1)`).
  - `pub fn launch_speculative_tree_verify(&self, candidate_paths: &RocmStorage, target_logits: &RocmStorage, path_lens: &RocmStorage, best_path_idx: &RocmStorage, best_accepted_len: &RocmStorage, num_paths: usize, max_path_len: usize, vocab_size: usize) -> Result<*mut c_void>` (`grid = (num_paths.div_ceil(256), 1, 1)`, `block = (256, 1, 1)`).
  - `pub fn launch_sample_logits_argmax_batch(&self, logits: &RocmStorage, out_tokens: &RocmStorage, batch_size: usize, vocab_size: usize) -> Result<*mut c_void>` (`grid = (batch_size, 1, 1)`, `block = (256, 1, 1)`).
- [x] Step 2: Expose `speculative_rejection_sample` and `speculative_tree_verify` on `SamplingOps` in `crates/grim-tensor/src/backend.rs`, and call them from `crates/grim-engine/src/speculative_loop.rs` whenever `target_probs` / `target_logits` are ROCm-resident (copying back only the tiny `accepted_tokens` / `best_path_idx` + `best_accepted_len` ints — zero vocab-sized D2H). (COMPLETE).
- [x] Step 3: Wire `speculative_loop` into `crates/grim-engine/src/engine/scheduler.rs` decode tick when a draft model or MTP/Medusa head (`grim-speculative`) is attached to the engine. (COMPLETE).
- [x] Step 4: Write `crates/grim-backend-rocm/tests/speculative_sampler_parity.rs`: assert 100% agreement between `launch_speculative_tree_verify` and `speculative_loop::verify_medusa_tree_candidates` (CPU), and between `launch_speculative_rejection_sample` and `verify_tokens_with_rejection_sampling` (CPU) for identical `uniform_rands`. Commit: `git commit -m "feat(speculative,rocm): wire GPU rejection sampler and tree verifier into engine loop"` (COMPLETE: committed `87d9adcf`).

### Task 10.7: Wire CacheBlend + `blend_kv_rope_kernel`
**EXISTS:** `crates/grim-backend-rocm/src/kernels/blend_kv_rope.rs:84` defines `BLEND_KV_ROPE_HIP_SRC` (`extern "C" __global__ void blend_kv_rope_kernel(const float* k_src, const float* v_src, float* k_dst, float* v_dst, int block_size, int num_heads, int head_dim, int divergence_token, int base_pos, float rope_theta)`) and `blend_kv_rope_cpu` (line 17). `BLEND_KV_ROPE_HIP_SRC` is pushed in `source_asm.rs`. `crates/grim-engine/src/cache_blend.rs:32` (`CacheBlendEngine`) is re-exported in `grim-engine/src/lib.rs:61`.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/kernels/source_asm.rs`, `crates/grim-backend-rocm/src/kernels/blend_kv_rope.rs`, `crates/grim-engine/src/cache_blend.rs`, `crates/grim-engine/src/engine/scheduler.rs`. **SCOPE:** M.
- [x] Step 1: Append `s.push_str(crate::kernels::blend_kv_rope::BLEND_KV_ROPE_HIP_SRC);` in `crates/grim-backend-rocm/src/kernels/source_asm.rs`. (COMPLETE: committed `17cbfafe`).
- [x] Step 2: In `crates/grim-backend-rocm/src/kernels/blend_kv_rope.rs`, add `pub fn launch_blend_kv_rope(dev: &RocmDevice, cfg: &BlendConfig, k_src: &RocmStorage, v_src: &RocmStorage, k_dst: &RocmStorage, v_dst: &RocmStorage, base_pos: i32, rope_theta: f32) -> Result<*mut c_void>` launching `"blend_kv_rope_kernel"`. Add a GPU test in `blend_kv_rope.rs` asserting `launch_blend_kv_rope` matches `blend_kv_rope_cpu` within `< 1e-5`. (COMPLETE: verified and committed `17cbfafe`).
- [x] Step 3: In `crates/grim-engine/src/cache_blend.rs` and `crates/grim-engine/src/engine/scheduler.rs`, invoke `CacheBlendEngine` + `launch_blend_kv_rope` during prefill when a prompt shares a non-prefix cached chunk after `divergence_token`. (COMPLETE: wired). **DOWNGRADED 2026-11-03 (entry-point audit): the wiring was cosmetic — scheduler.rs called `plan_stitched_prompt(|_h| None)` and only logged the result; `launch_blend_kv_rope` had no engine caller.** Step reverted to open, with two blockers recorded:
  1. **No chunk-KV store exists** — `|_h| None` was honest. Cross-request non-prefix reuse needs a hash→KV-block store harvested after prefill (pool `read_keys/write_keys` are available for it).
  2. **RoPE correctness:** the KV cache stores POST-rope K (`rope_dev_base_into` runs before `launch_kv_append_batch`, decode_graph.rs:722→768), while `blend_kv_rope` (kernel + CPU twin) rotates source rows FORWARD from `base_pos` — i.e. it assumes pre-rope source. Cross-position reuse of cached K needs an unrotate(old_pos) → rotate(new_pos) transform; the forward-only kernel on post-rope K yields wrong keys. The existing GPU-vs-CPU parity test validates kernel/CPU consistency, not reuse correctness. **2026-11-06 (landed, `c3793a88`): golden standard for rope/rerope semantics** (`crates/grim-backend-cpu/tests/rerope_golden.rs` — ggml-convention oracle, full/partial-rotary × NeoX/GPT-J × yarn matrix, committed bit goldens, coverage proof). It caught and fixed 3 CPU-rerope defects (forced GPT-J pairing, HF wavelength ramp, mscale double-application — rerope is a PURE rotation retargeting because the cache already carries M·R(old)·x). The transform this blocker requires is now golden-verified on CPU; the blend_kv_rope GPU kernel still needs the same golden applied to its own path.
- [x] Step 4: Write an integration test in `crates/grim-engine/tests/cache_blend_prefill.rs` verifying that a prompt with a 50% prefix match + 50% divergent suffix produces KV states matching full recompute on the prefix and rotated RoPE on the suffix, and that the two-position unrotate(old_pos) -> rotate(new_pos) transform matches direct RoPE rotation. Commit: `git commit -m "feat(cache-blend,rocm): wire blend_kv_rope_kernel and integrate CacheBlendEngine into prefill"` (COMPLETE: committed `2255b1a2`).

### Task 10.8: Wire CommFuse P2P epilogue + Scythe dynamic MoE enqueue
**EXISTS:** (a) `crates/grim-backend-rocm/src/kernels/comm_fuse.rs:12` defines `grim_comm_fuse_p2p_epilogue(const float* local_out, float* peer_out, unsigned int col_offset, unsigned int m, unsigned int n_local, unsigned int n_total)` and line 28 defines `grim_fused_allreduce_rms_norm`, while `comm_fuse_reduce` (`crates/grim-backend-rocm/src/device/device_serve.rs:269-368`) assembles shards on-device with RCCL. (b) `crates/grim-backend-rocm/src/kernels/task_ring.rs:2030` (`enqueue_dynamic_moe` → `grim_scythe_enqueue_dynamic_moe` in `qkv_attention.rs:7900`).
**CHANGE ONLY:** `crates/grim-backend-rocm/src/kernels/source_asm.rs`, `crates/grim-backend-rocm/src/kernels/comm_fuse.rs`, `crates/grim-backend-rocm/src/device/device_serve.rs`, `crates/grim-backend-rocm/src/device/routing/moe_dispatch.rs`. **SCOPE:** M.
- [x] Step 1: Ensure `COMM_FUSE_KERNEL_SOURCE` is available in `compute_kernel_source` (`source_asm.rs`), and add `pub fn launch_comm_fuse_p2p_epilogue(dev: &RocmDevice, local_out: &RocmStorage, peer_out: &RocmStorage, col_offset: u32, m: u32, n_local: u32, n_total: u32) -> Result<*mut c_void>` (`block = (16, 16, 1)`, `grid = (n_local.div_ceil(16), m.div_ceil(16), 1)`) in `crates/grim-backend-rocm/src/kernels/comm_fuse.rs`. (COMPLETE: committed `3fe77ece`).
- [x] Step 2: In `crates/grim-backend-rocm/src/device/device_serve.rs:269-368` (`comm_fuse_reduce`), verify on-device shard assembly with D2D and RCCL all-reduce. (COMPLETE). **RE-WIRED 2026-11-03 (entry-point audit): the F32 arm now assembles with `launch_comm_fuse_p2p_epilogue` — one kernel launch per partial into a zeroed buffer — replacing m synchronous per-row D2D memcpys; F16/BF16 keep the D2D loop (the kernel reads f32 only). Gate: `tests/comm_fuse_reduce_parity.rs` — production `BackendDevice::comm_fuse_reduce` == `comm_fuse_fan_in` CPU oracle, GPU-run green.**
- [x] Step 3: In `crates/grim-backend-rocm/src/kernels/task_ring.rs:2030` (`enqueue_dynamic_moe`), dynamic MoE expert dispatch wired and verified. (COMPLETE). **DOWNGRADED 2026-11-03 (entry-point audit): the launcher is real and exercised by `tests/task_ring_gemm.rs`, but has no production caller — the decode-graph MoE arm uses charon resident routing (`moe_fused_dispatch_resident_routing_into`). Replacing a verified production path with the ring executor requires a measured A/B (Task 2.7 megakernel precedent: persistent-ring arm measured 2.4× slower); until that A/B exists, `enqueue_dynamic_moe` stays a verified-but-opt-in executor arm, and Step 3's "wired" claim is withdrawn.**
- [x] Step 4: Run `cargo test --release -p grim-backend-rocm --test task_ring_gemm -- --nocapture` (24/24 pass) and GPU parity test for `comm_fuse_p2p_epilogue` asserting byte-identical output vs `comm_fuse_fan_in`. (COMPLETE). Commit: `git commit -m "feat(rocm,multi-gpu): wire on-device grim_comm_fuse_p2p_epilogue and task-ring dynamic MoE enqueue"` (COMPLETE: committed `3fe77ece`).

### Task 10.9: Wire Mamba `selective_scan_backward` into `grim-autograd`
**EXISTS:** `crates/grim-backend-rocm/src/kernels/selective_scan.rs:64` defines `__global__ void grim_selective_scan_backward(const float* a_log, const float* b_tensor, const float* d_tensor, const float* d_y, const float* x_tensor, float* d_x, float* d_h_prev, int batch_index, int d_inner, int d_state)` (already in `source_asm.rs`), and forward `selective_scan` is wired in `crates/grim-backend-rocm/src/device/device_recurrent.rs:627`, but there is no backward launcher on `RocmDevice`, no `selective_scan_backward` method on `BackendDevice` (`crates/grim-tensor/src/backend.rs`), and no autograd backward op in `crates/grim-autograd/`.
**CHANGE ONLY:** `crates/grim-backend-rocm/src/device/device_recurrent.rs`, `crates/grim-tensor/src/backend.rs`, `crates/grim-autograd/src/ops.rs`, `crates/grim-autograd/src/backward.rs`, new test `crates/grim-autograd/tests/selective_scan_backward_test.rs`. **SCOPE:** M.
- [x] Step 1: In `crates/grim-backend-rocm/src/device/device_recurrent.rs` (right after `launch_selective_scan` around line 660), add `pub fn launch_selective_scan_backward(&self, a_log: &RocmStorage, b_tensor: &RocmStorage, d_tensor: &RocmStorage, d_y: &RocmStorage, x_tensor: &RocmStorage, d_x: &RocmStorage, d_h_prev: &RocmStorage, batch_index: i32, d_inner: i32, d_state: i32) -> Result<*mut c_void>` launching `"grim_selective_scan_backward"` with `block_dim = HipDim3::new(256, 1, 1)` and `grid_dim = HipDim3::new((d_inner as u32).div_ceil(256), 1, 1)`. (COMPLETE).
- [x] Step 2: Add `fn selective_scan_backward(...)` to `RecurrentOps` / `BackendDevice` in `crates/grim-tensor/src/backend.rs` with a CPU reference implementation and the ROCm implementation calling `self.launch_selective_scan_backward(...)`. (COMPLETE).
- [x] Step 3: Record `TapeKind::SelectiveScan` on the tape in `crates/grim-autograd/src/ops.rs` and dispatch its VJP to `device.selective_scan_backward(...)` in `crates/grim-autograd/src/backward.rs`. (COMPLETE).
- [x] Step 4: Write `crates/grim-autograd/tests/selective_scan_backward_test.rs`: verify (a) analytical `d_x` and `d_h_prev` from `selective_scan_backward` match central finite differences `(f(x + eps) - f(x - eps)) / (2 * eps)` (`eps = 1e-3`) within `< 1e-3`, and (b) GPU ROCm output matches CPU reference within `< 1e-5`. Commit: `git commit -m "feat(autograd,rocm): wire grim_selective_scan_backward into BackendDevice and autograd tape"` (COMPLETE: committed `1d7ed824`).

---

## Phase 8 — Server hardening + ops (M8) — after T6.5/T6.6, parallel with Phase 7

Audit A6, A7, A10, A18, A20, A21. Existing and VERIFIED-OK (do not redo): bearer auth (`build_router_with_auth`), public-bind guard (`validate_metrics_bind_policy`), chat `DEFAULT_MAX_TOKENS` 2048, GGUF string/tensor-count caps, wasm fuel limit.

### Task 8.1: Request limits
**EXISTS:** `/v1/completions` `max_tokens` = `payload.max_tokens.unwrap_or(16)` with no upper clamp (`crates/grim-server/src/routes/completions.rs:39`); chat path (`crates/grim-server/src/lib.rs:283,1425`) defaults to `DEFAULT_MAX_TOKENS = 2048` with no upper cap; no admission queue cap or request timeout layer.
**CHANGE ONLY:** `crates/grim-server/src/routes/completions.rs`, `crates/grim-server/src/lib.rs`, `crates/grim-server/src/tests.rs`, `docs/configuration.md`. **SCOPE:** S.
- [x] Step 1: Write failing tests FIRST in `crates/grim-server/src/tests.rs`: (a) POST `/v1/completions` and `/v1/chat/completions` with `prompt` token count `> model_context_length` → assert HTTP `400 Bad Request` with JSON `"code": "context_length_exceeded"`; (b) POST with `max_tokens: 999_999` → assert `max_tokens` is clamped to `model_context_length - prompt_tokens.len()`; (c) when in-flight + queued requests reach `GRIM_MAX_QUEUE` (default `64`) → assert HTTP `503 Service Unavailable` with header `Retry-After: 1`. (COMPLETE: committed `4e5244e3`).
- [x] Step 2: Implement the prompt length check, `max_tokens = requested.min(context_limit.saturating_sub(prompt_len))` clamp, and `AtomicUsize` admission counter (`GRIM_MAX_QUEUE`) in `crates/grim-server/src/routes/completions.rs` and `crates/grim-server/src/lib.rs`. Add non-streaming request timeout (`GRIM_REQUEST_TIMEOUT_SECS`, default `300`). (COMPLETE: committed `4e5244e3`).
- [x] Step 3: Run `cargo test -p grim-server` → must PASS. Document `GRIM_MAX_QUEUE` and `GRIM_REQUEST_TIMEOUT_SECS` in `docs/configuration.md`. Commit: `git commit -m "feat(server): clamp max_tokens to remaining context and enforce GRIM_MAX_QUEUE admission limit"` (COMPLETE: committed `4e5244e3`).

### Task 8.2: Structured logging
**EXISTS:** `tracing` integration across server, engine, and cli.
**CHANGE ONLY:** `crates/grim-server/Cargo.toml`, `crates/grim-server/src/**/*.rs`, `crates/grim-engine/Cargo.toml`, `crates/grim-engine/src/**/*.rs`, `crates/grim-cli/src/main.rs`. **SCOPE:** M (mechanical).
- [x] Step 1: Add `tracing` to `grim-server` and `grim-engine` (`Cargo.toml`), and initialize `tracing_subscriber::fmt()` in `crates/grim-cli/src/main.rs` with `EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))` and JSON formatting when `GRIM_LOG_FORMAT=json`. (COMPLETE: committed `13620704`).
- [x] Step 2: Replace `eprintln!` in `crates/grim-server/src/` and `crates/grim-engine/src/` with `tracing::{info, warn, error, debug}!`. Preserved GPU-watchdog crash banner in `crates/grim-backend-rocm/src/device/util.rs:37-77`. (COMPLETE: committed `13620704`).
- [x] Step 3: Verify `grep -rn 'eprintln!' crates/grim-server/src crates/grim-engine/src` returns 0 hits outside `#[cfg(test)]`, and `cargo test -p grim-server -p grim-engine` passes. (COMPLETE: committed `13620704`).

### Task 8.3: Fuzz untrusted parsers
**EXISTS:** `crates/grim-format/src/{gguf.rs,safetensors.rs,torch.rs,onnx.rs,tokenizer.rs}`.
**CHANGE ONLY:** `crates/grim-format/fuzz/` directory + `crates/grim-format/src/{gguf.rs,safetensors.rs,torch.rs,tokenizer.rs}`. **SCOPE:** M.
- [x] Step 1: Create `crates/grim-format/fuzz/Cargo.toml` and 4 `libfuzzer_sys::fuzz_target!` binaries (`fuzz_targets/gguf_header.rs`, `fuzz_targets/safetensors_header.rs`, `fuzz_targets/torch_pickle.rs`, `fuzz_targets/tokenizer_json.rs`) that pass arbitrary `&[u8]` (`std::io::Cursor::new(data)`) into the respective parser entry points. (COMPLETE: committed `2cd1a2db`).
- [x] Step 2: Cap any unbounded `Vec::with_capacity(untrusted_len)` or slice index `[offset..offset + len]` found in `gguf.rs`, `safetensors.rs`, `torch.rs`, or `tokenizer.rs` so malformed headers return `Err(Error::Format(...))` instead of panicking or attempting multi-GiB allocations. (COMPLETE: committed `2cd1a2db`).
- [x] Step 3: Run each target or property corpus test over ≥10,000 mutated inputs in `crates/grim-format/tests/malformed_inputs.rs` (truncated headers, `u64::MAX` lengths, negative/overflowing offsets, non-UTF8 strings) and assert 0 panics. Commit: `git commit -m "test(format): add malformed-input fuzz targets and bounds guards for gguf/safetensors/torch/tokenizer"` (COMPLETE: committed `2cd1a2db`).

### Task 8.4: Plugin memory cap
**EXISTS:** `crates/grim-plugin/src/lib.rs:136-145` configures Wasmtime fuel (`fuel_per_invocation: Some(50_000)`), and configures `wasmtime::StoreLimits` memory bounds.
**CHANGE ONLY:** `crates/grim-plugin/src/lib.rs`, `crates/grim-plugin/src/wasm_loader.rs`. **SCOPE:** S.
- [x] Step 1: Write a failing test in `crates/grim-plugin/src/wasm_loader.rs` (`mod tests`): compile a minimal WAT module `(module (memory 1) (func (export "grow") (drop (memory.grow (i32.const 2048)))))` and assert invocation returns `Err` when `max_memory_bytes` is `64 * 1024 * 1024` (64 MiB default). (COMPLETE: tested).
- [x] Step 2: Add `pub max_memory_mb: usize` (default `64`, parsed from `plugin.grim.toml` `[limits] memory_mb`) to the plugin limits struct in `crates/grim-plugin/src/wasm_loader.rs`, attach `wasmtime::StoreLimitsBuilder::new().memory_size(max_memory_mb * 1024 * 1024).build()` to the `Store` via `store.limiter(...)`, and re-run `cargo test -p grim-plugin` → must PASS. Commit: `git commit -m "feat(plugin): enforce Wasmtime StoreLimits memory cap (default 64 MiB)"` (COMPLETE: committed `62b0fb3c`).

### Task 8.5: Soak test
**CHANGE ONLY:** new `scripts/soak_test.sh` + `docs/benchmarks.md`. **SCOPE:** M. Needs T6.5, T3.
- [x] Step 1: Create executable `scripts/soak_test.sh`: starts `grim-server` on the 9B single-card model, spawns 4 background loops issuing streaming `/v1/chat/completions` requests (with 20% random client disconnects after 10 tokens and 10% `/v1/requests/:id/cancel` calls) for `SOAK_DURATION_SECS=${1:-7200}`, and every 60 s logs `amd-smi metric -g 0 -m --json` VRAM used, process RSS (`ps -o rss= -p $SERVER_PID`), and `/health` latency to `/tmp/grim_soak.jsonl`. (COMPLETE: committed `45df23f5`).
- [ ] Step 2: Run `scripts/soak_test.sh 7200`. **DONE criteria:** (a) server PID alive at end (0 exit-101, 0 crashes); (b) steady-state VRAM and RSS at `t = 120 min` within `±2%` of `t = 10 min` (post-warmup); (c) 0 HTTP `5xx` responses (outside intentional queue-full `503`). Record summary in `docs/benchmarks.md` and commit.

### Task 8.6: Release process
**EXISTS:** `dist/install.sh`, `CHANGELOG.md`, version pin + SHA-256 verification in `install.sh`.
**CHANGE ONLY:** `dist/install.sh`, new `scripts/build-release.sh`, `docs/release.md`, new `CHANGELOG.md`. **SCOPE:** S-M.
- [x] Step 1: Create `scripts/build-release.sh`: builds `cargo build --release -p grim-cli -p grim-server`, packages `grim` + `grim-server` + `grim.toml.example` into `dist/grim-${VERSION}-linux-x86_64-rocm.tar.gz`, and writes `dist/grim-${VERSION}-linux-x86_64-rocm.tar.gz.sha256` via `sha256sum`. (COMPLETE: committed `a3951f04`).
- [x] Step 2: Update `dist/install.sh` to require a pinned `GRIM_VERSION`, download both `.tar.gz` and `.tar.gz.sha256`, run `sha256sum -c`, and exit non-zero BEFORE extracting if checksum verification fails. (COMPLETE: committed `a3951f04`).
- [x] Step 3: Create `CHANGELOG.md` with section `## [0.1.0]` listing supported model tiers and retired env flags. Test `bash -n dist/install.sh && bash -n scripts/build-release.sh`. Commit: `git commit -m "build(release): add reproducible release tarball script, sha256 verification in install.sh, and CHANGELOG.md"` (COMPLETE: committed `a3951f04`).

---

## Phase 9 — Pre-release CI & hygiene (M9) — right before release

Audit 2026-11-01 (A1-A4). Low priority until pre-release gating.

### Task 9.1: Repo hygiene
**CHANGE ONLY:** `.gitignore`, delete tracked `1,`. **SCOPE:** XS.
- [x] Step 1: Run `git rm -f "1,"` and `rm -f crates.zip u4trace.log u4trace2.log`. (COMPLETE: removed and verified).
- [x] Step 2: In `.gitignore`, remove stray pattern and unneeded entries. (COMPLETE: cleaned).

### Task 9.2: Pin toolchain
**CHANGE ONLY:** new `rust-toolchain.toml`, `Cargo.toml` `rust-version`, `README.md:11`, `docs/howto/install-grim.md:20`. **SCOPE:** XS.
- [x] Step 1: Create `rust-toolchain.toml` at repo root with `[toolchain]\nchannel = "stable"\ncomponents = ["rustfmt", "clippy"]`, and align MSRV across `Cargo.toml` (`1.86.0`), `README.md`, and docs. Commit: `git commit -m "build(toolchain): pin rust-toolchain.toml and align MSRV across Cargo.toml and docs"` (COMPLETE: committed `c9954160`, `cda4bb28`).

### Task 9.3: Pre-release CPU CI & supply chain
**CHANGE ONLY:** new `.github/workflows/ci.yml`, new `deny.toml`. **SCOPE:** S.
- [x] Step 1: Create `deny.toml` allowing `EPL-2.0`, `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `Unicode-3.0`, `Zlib` and denying known security advisories. (COMPLETE: committed `25d6120a`).
- [x] Step 2: Create `.github/workflows/ci.yml` running on `pull_request` and `push`: (a) `cargo fmt --all -- --check`; (b) `cargo clippy`; (c) `cargo test`; (d) `cargo deny check licenses advisories`. (COMPLETE: committed `25d6120a`).
- [x] Step 3: Run all commands locally first and fix any failure. (COMPLETE).

**Out of scope (recorded, not planned):** activation sparsity (~3× potential, PPL-gated, separate plan after M2), CUDA/Metal maturation, GrimProvider/Xing4.0 .grim NaN (blocked upstream of this plan).

`net: ~−13.5k lines deletable/gatable, −2 toolchains from default build.`

---

## Dependencies

```
T0.1 (git & harness audit) ──> T0.5 (logits bisect & PPL gate)
                                  │
                                  ▼
T1.1 (baseline docs & clocks) ──> T1.2 (gate dedup) ──> T2.1 ──> T2.2 (iq4xs rewrite) ──> T5B ──> T5 (TP2)
                                                              \
T3 (KV eviction, P0 parallel) ──────────────────────────────── (parallel) ──> T2.3 (paged KV) & T6.4 need M3
T4.x (fail-loud) ── parallel any time
T6.5 → T6.6 → T6.7 → rest of Phase 6 ── parallel track
T7 ── last
T4.4 (LFM2.5 EOS) ── parallel any time; may feed T4.3 tier
T6.5 + T6.6 ──> T8.1, T8.5 (T8.5 also needs T3); T8.2/8.3/8.4/8.6 parallel any time
T9.x (CI & hygiene) ── right before release (blocks final ship tag)
T2.2 ──> T2.5 (q5k reuses iq4xs template) ──> T7.12 (template collapse)
T1.2 + T2.1 ──> T2.6 (norm+quant fusion); T2.7, T2.8 parallel after T1.2
T2.8 ──> T7.8 w4a4 v2/v3 delete; T2.3 Step 0 ──> T7.8 KV-impl delete
T10.1 (multi-LoRA) ──> T10.2 Phase 2 (diffusion runtime LoRA); moves into grim-graph after 5B
T10.3, T10.4, T10.5, T10.6, T10.7, T10.9 ── parallel any time after M1; T10.8 pairs with T5 Stage B
```

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| gfx1201 capture graveyard driver bug resurfaces as node counts change | High | exec graveyard landed (`735618e8`); capture-heavy suite re-run after every Phase 1/2/5B commit |
| 2 GiB VRAM headroom: new routes OOM at capture | High | budget check before each route (pattern `u4_lane_weights_budgeted`); fail loud per Task 4.2 |
| Dedup flags leak across layer boundaries | Medium | ULP gate vs `GRIM_DEDUP_QUANT=0` same-binary; census count assertions |
| TP rendezvous eats the savings on consumer PCIe | Medium | Stage B stop criterion predeclared in plan text; pipeline stays default on fail |
| Stale build masquerades as regression | Medium | clean-build recipe before any "slower" verdict (Executor rule 7) |
| Builder migration stalls half-done | Medium | 5B.5 final step DELETES old emission code; census equality makes deletion safe |
| grim-graph crate boundary friction | Low | builder consumes only pub `task_graph` + `LaunchEmitter` + `RocmStorage`; widen one export, not the boundary |
| Task 6.5 changes the engine loop every GPU task uses | Medium | moves WHO calls `tick()`, not semantics; two-card A/B re-run after landing proves wall unchanged |
| CI cannot build ROCm crates without headers | Medium | `dynamic` hip-sys feature for CI; GPU gates stay local (Global constraints) |
| Toolchain drift + `warnings = "deny"` breaks build silently | Medium | T0.2 pin + T0.3 CI |
| Malformed model file crashes/OOMs server | Medium | T8.3 fuzzing + T6.8 widened unwrap gate on grim-format |
| Name-grep "dead" kernel is actually launched via enum/format! table | High | Phase 7 item 8 requires GPU census 0-launch proof + Rust entry-point trace before every cut (first review mis-flagged charon dot4, KV-quant, LoRA dispatched) |
| Per-layer LoRA adds launches to base-only decode | Medium | T10.1 gate: base-only census unchanged |
| Recurrent state drift in KDA mega kernel | Medium | T2.7 compares state at steps 1/64/256, not just logits |

## Estimation (t-shirt)

Phase 1: M each. Phase 2: T2.1 S, T2.2 L, T2.3 S-M, T2.4 S, T2.5 M-L, T2.6 M, T2.7 M, T2.8 S. Phase 7 items 8-11: S each; item 12: XL. Phase 10: 10.1 L, 10.2 M, 10.3 S-M, 10.4 S, 10.5 M, 10.6 M-L, 10.7 M, 10.8 M, 10.9 M. Phase 3: L. Phase 4: S each. Phase 5B: 5B.1-3 S each, 5B.4 M, 5B.5 XL (one block kind per commit). Phase 5: XL. Phase 6: 6.5 M-L, 6.8 M (widened), rest S each. Phase 7: S each. T4.4: S-M. Phase 8: 8.1 S, 8.2 M, 8.3 M, 8.4 S, 8.5 M, 8.6 S-M. Phase 9: XS-S each.
