# Avalon -> grim transfer review: TTFT + decode speed for Qwen3.8-27B / Qwen3.8-Flash-Next

Review date: 2026-11-20. Reviewer: ZCode. Status: COMPLETE (three Avalon code-verification agents + three PLAN-astrum code-verification agents delivered; see Appendices B, C, D).

## Scope and method

- Source under review: `https://github.com/itekdev/llm-Avalon` (mirror of `freriadoc/Avalon`), cloned at
  `/drive/bigfast/grim/old/repo/avalon`, HEAD `26a2155`, depth-1 + GitHub API history for 500 commits.
- Baseline: grim at `/drive/bigfast/grim`, HEAD `9aa53b9a` (clean tree, branch `main`).
- Avalon = llama.cpp fork (base up to 2026-10-31) tuned for ONE model: Qwen3.8-Flash-Next
  (`qwen4exp`, 512 experts / 10 active / 48 layers / GDN + QSA hybrid) on consumer RDNA3/ROCm with experts in host RAM.
- Method: (1) isolate Avalon-specific delta vs upstream llama.cpp by commit history + symbol presence check;
  (2) read implementation code for each candidate; (3) check grim's current code for an equivalent or an open gap;
  (4) rank by measured effect, transfer cost, and overlap with existing grim plans. No phantom claims: every finding
  needs an in-tree anchor on both sides.
- Skills applied: `ai-inference-expert`, `ai-llm-inference` (roofline / TTFT-ITL framing, parity gates),
  `caveman-mode` (terse style), `ponytail` (reuse-first, smallest transfer that works).

## Evidence appendix A: Avalon commit-level claims (verified via GitHub API, 2026-11-20)

Quantified claims are from the fork's own commit messages (author measurements; not independently re-benchmarked here).
Path lists are from the same API responses. Symbols were confirmed present in the clone.

| ID | Avalon commit | Mechanism | Avalon's measured effect | Files touched |
|----|---------------|-----------|--------------------------|---------------|
| A1 | `26a2155` | grouped MoE SwiGLU rows along grid x | (HEAD, latest tuning) | `mmq.cu` |
| A2 | `07dcefe8` | context checkpoint state -> pinned buffer, copy after first token | server TTFT short 42-43 -> 27 ms; 19k 45 -> 29-34; 131k 57 -> 40-43 | `common/common.{cpp,h}`, `llama-context.{cpp,h}`, `server-context.cpp` |
| A3 | `f5a34ece` | summed MoE down + QSA slice=4096 ON by default | compute buffer 2750 -> 2191 MiB at -ub 9536 (less expert-cache eviction); after-19k 43.8/48.2 -> 51.9/52.2 t/s | `llama-graph.cpp`, `qwen4exp.cpp` |
| A4 | `3e892815` | `top_k_radix_row`: 3 radix passes, compaction, no per-chunk barrier | kernel [32768,2048] 2.25 -> 0.97 ms; [32768,9536] 10.6 -> 4.5 ms; prefill 19k 48.4 -> 42.0 ms (of 8.6 s) | `top-k.cu` |
| A5 | `043ffd06` | prompt expert ring default on HIP, 900 MiB lent | ~10% fewer entries moved/evicted per loan vs 1800 MiB | `ggml-cuda.cu`, `moe-cache.cu`, `moe-ring.h`, `expertringtest.cpp` |
| A6 | `e09e775f` | `LLAMA_MOE_DOWN_SUM` via summed MMQ (weighted atomic adds) | absorbed by hc K1 fusion | `ggml-cuda.cu`, `llama-graph.cpp` |
| A7 | `7cd93b54` | borrowed compute buffers in one chunk + growth margin | avoids doubling | `ggml*.c` |
| A8 | `90514ffa` | grouped MoE gate+up in one launch, tile prefixes per layer | 4090 prefill 131k +1.0-1.8%; PPL identical | `mmq.cu` |
| A9 | `c3b6e6f8` | MTP draft prompt pass from device rows (no host round trip) | 131k prefill +1.6-1.7%; VRAM flat | `llama-context.cpp` |
| A10 | `3d5aa1fc` / `367cf8b1` | fused `hc_post`+norm, image + inject written by producer | 19k: 149/186 boundaries fused; RX 7900 prefill -0.5% (link-limited); 4090 convert_unary 0.79 s / 1222 launches removed; bit-identical outputs | `dsv4-hc.{cu,cuh}`, `norm.{cu,h}`, `ggml-cuda.cu` |
| A11 | `e681cfd1` | `hc_pre`/`hc_post` thread per (column,token), 32-bit indices, residual read once | prefill +1.5%; generation unchanged | `dsv4-hc.cu` |
| A12 | `99dc4dd9` | multi-task kernel only for batches <=64 tokens | prefill +3.7%; GDN layer 18 -> 8.5 ms; PPL unchanged | `ggml-cuda.cu` |
| A13 | `e399c4cd` | QSA tile mask every prompt batch + 1 block/row top-k(64+) | 4090 prefill 19k +9.8%, 131k +2.7%; top-k kernel 1002 -> 604 ms | `top-k.cu`, `qwen4exp.cpp` |
| A14 | `2c346ba1` | QSA attention by tile lists; q8_0/q4_0 K/V gathered quantized | 262k cells/9536 q: 140.2 -> 127.9 ms dequant; head-group tile order -5.6/-6.5% | `fattn*.{cu,cuh}`, `qsa-select.cu`, `ggml.h`, `ggml.c` |
| A15 | `d66513f4` | one flash attention per QSA layer (tile lists) vs one per slice | 250k prompt graph time 183.5 -> 170.2 s; QSA attn 28.0 -> 18.8 s | `models.h`, `qwen4exp.cpp` |
| A16 | `d71749e8`, `9b1c7a48`, `876c9228` | QSA slices / 8-query tiles / all-batch tile mask (switches) | off by default; basis for A13-A15 | `qwen4exp.cpp`, `top-k.cu` |
| A17 | `e2b9b600` | HIP GDN recurrence in 16-token chunks on RDNA3 matrix cores (Strata kernels) | layer at 9536 tok: 3.27 ms vs ~9 ms; prefill 1548-1555 -> 1607-1612 t/s (+3.4%); PPL ~flat | `gdn-wmma.{cu,cuh}`, `gated_delta_net.cu`, `ggml-cuda.cu` |
| A18 | `ad2c6e75` | cached ssm_conv for prompt batch, chunks of 16 | prefill 1634-1644 -> 1662-1670 t/s (+2.0%); PPL ~flat | `ssm-conv.cu`, `ggml-cuda.cu` |
| A19 | `76c3c643` / `b7930400` | GDN on mma.sync tensor cores (CUDA only; not built for HIP) | 4090 prefill +2.2-3%; HIP unchanged | `gdn-wmma.cu` |
| A20 | `e66c5be7` | chunked GDN with deferred commit (MTP replay window) | fixes chunked GDN never running in server with MTP (4090: 540 scalar launches, 2.57 s) | `gated_delta_net.cu`, `gdn-wmma.cu` |
| A21 | `7fc48274` | backend greedy: no chain filters, no logits output; two-stage argmax of 248k rows | added kernels/step 0.37 -> ~0.02 ms; argmax 39.5 us for 248320 cols | `sampling.{cpp,h}`, `argmax.cu`, `llama-sampler.cpp`, `server-context.cpp` |
| A22 | `a3870717` | one shared backend greedy chain for all requests | removes 155-178 ms worst-case graph re-reservation; prompt 322-370 vs 277-395 ms (noise), step 22.35-23.01 vs 22.48-23.33 ms | `sampling.cpp`, `llama-context.{cpp,h}`, `llama-sampler.{cpp,h}` |
| A23 | `1e193158` | backend sampler inputs in own buffer | kept graphs can replay again | `sampling.cpp`, `llama-context.cpp` |
| A24 | `510abcb1` | MoE tail skip (Strata `STRATA_ROUTE_TAIL_SKIP`, rank 7+) | MTP best 91.6 -> 99.6 t/s; after 19k 78.5 -> 88.5; top-1 98.6-99.7%, KL 0.002-0.004 (LOSSY) | `moe-cache.cu` |
| A25 | `853981b4` | quantized embedding rows crossed raw, dequant on GPU | Q8_0 token embd 10240 -> 2720 B/row; Q5_1 PLE table 10240 -> 1920 B/token | `ggml*.{c,h,cpp}`, `llama-graph.{cpp,h}`, `qwen4exp.cpp` |
| A26 | `c49a469f`, `4884cd5a` | PLE table read once into owned RAM (O_DIRECT, mapped address) | avoids page-cache eviction; load-time | `llama-context.cpp` et al. |
| A27 | `5942fb7b`, `b7eacfc6`, `043ffd06` | prefill expert byte ring (GGML_SCHED_PREFETCH_RING_MB / MOE_RING_PREFILL=900, default on HIP) | 4->8 records; no host wait; ~10% fewer evictions; simulator 4000 seeds ASan/UBSan | `ggml-backend.cpp`, `mmq.cu`, `ggml-sched-ring.h`, `moe-ring.h`, tests |
| A28 | `29e013b7` | ubatch fitted to what expert cache can lend now (`LLAMA_UBATCH_FIT`) | -ub 9536: 2559 MiB; -ub 32768: 8198 MiB (prompt in one ubatch) | `llama-batch.{cpp,h}`, `llama-context.{cpp,h}`, `moe-cache.cu` |
| A29 | `a40bb4f5`, `56581776`, `9e9cbb83` | 2-GPU layer pipeline: prompt pipelining, hidden-state-only crossing, full ubatch for long prompts | 2x A4000 60k prefill 1649 -> 1761 t/s; 120k 1451 -> 1598 (4 calls -> 1); generation unchanged | `llama-context.cpp`, `speculative.cpp` |
| A30 | `03c749ef`, `213b569d`, `9db58a10` | expert arena in registered host RAM (O_DIRECT, LRU), GPU driver registration limit | 16/32 GB RAM: 13-17 -> 49-88 t/s in README table (lossy scenario) | `moe-cache.cu`, `moe-ram-budget.h` |
| A31 | `510abcb1`+README | hosted MoE: GPU cached experts + CPU misses, 1-2 moves/GPU, inside HIP graph. **See Appendix C2: `--moe-cache-cpu-overlap` parsed but inert**; split/protocol live | 44% faster than Strata short prompt with MTP (README, headline) | `moe-cache.cu` et al. |
| A32 | `ad4ff656` | answer_room: close think phase early to leave answer room | GSM8K 7 misses -> 0 cut-off; ms/token unchanged (10.754 vs 10.746) | `reasoning-budget.{cpp,h}`, `sampling.cpp`, `server-common.cpp` |

Avalon-specific (absent from upstream llama.cpp at `old/repo/llama.cpp-master`, checked by symbol grep 2026-11-20):
`qsa_tile`, `qsa-select.cu`, `MOE_RING_PREFILL`, `MOE_TAIL_SKIP`, `MOE_DOWN_SUM`, `GGML_CUDA_MOE_ARENA`,
`LLAMA_STAGES`, `answer_room`, custom `top_k_radix_row` (upstream has a SYCL top-k only).

## Transfer findings (ranked)

Verdict legend: **PULL** = net-new and worth building; **WIRE** = grim already has the asset, only production wiring missing;
**PARTIAL** = useful sub-idea after adaptation; **NO** = already covered / closed / wrong architecture / wrong backend.
Model column: `FN` = Qwen3.8-Flash-Next (`qwen4exp`), `27B` = Qwen3.8-27B, `both` = engine-level.

Grim anchors verified at HEAD `9aa53b9a` (clean tree). Avalon anchors are commit IDs from appendix A.

### Tier 1 - highest value, mostly Flash-Next TTFT

1. **QSA tile-list sparse attention + device indexer + gathered quantized K/V** - `FN`, long-context TTFT. **WIRE + PORT**.
   - Avalon: A13/A14/A15: dense `[n_kv,n_t]` mask replaced by `ggml_qsa_select_tiles` (F16 `[n_list,16,n_tiles]` mask +
     per-tile sorted cell list + live counts), one sparse MMA readout per tile, q8_0 K / q4_0 V gathered and dequantized on
     load; 250k prompt graph 183.5 -> 170.2 s, QSA attention 28.0 -> 18.8 s; 4090 prefill +9.8% @19k.
   - grim: device kernels `grim_kv_gather_q81`, `grim_qsa_index_score`, `grim_kv_gather_f32` exist
     (`crates/grim-backend-rocm/src/kernels/qkv_attention.rs:5987-6097`, launchers `:7670,7731,7781`) but **only test
     callers** (`tests/qsa_gather_parity.rs:146,236,351`). Production `qwen4exp_flash_next.rs:1504-1710` copies K/V to host
     (`to_vec_f32` :1534-1537,1562-1564), builds mask on host (`qwen4exp_qsa.rs:175,276,304`), runs host attention then
     `move_to_device` (:1669-1710); long prefill falls back dense (:1695-1710). Single-sequence only (:1540-1543).
   - Transfer: stage 1 = wire existing `launch_qsa_index_score` + `launch_kv_gather_f32` into `:1504-1710`, keep host mask
     as oracle, gate on `GRIM_QWEN38_QSA`; stage 2 = port Avalon's tile-list mask/gather kernel (`qsa-select.cu`,
     `fattn-mma-f16.cuh`) to HIP. HIPRTC rule: append new kernels at END of shared `KERNEL_SOURCE`.
   - Gate: extend `qwen4exp_qsa_wiring_test.rs` + `qsa_gather_parity.rs`; token identity vs current host path on real
     checkpoint (Gate D token-31 near-tie flake -> rerun before reporting failure).

2. **GDN WY-form chunked recurrence on matrix cores, including MTP deferred-commit** - `FN`, prefill TTFT. **WIRE then PORT**.
   - Avalon: A17 (`e2b9b600`: RDNA3 WMMA, 16-token chunks, WY form, T/P prep + scan with state in WMMA accumulators,
     layer at 9536 tok 3.27 ms vs ~9 ms, prefill +3.4%, PPL flat), A20 (`e66c5be7`: deferred-commit window replay first -
     chunked kernels never ran with MTP before: 540 scalar launches / 2.57 s), A18 (`ad2c6e75`: cached ssm_conv 16-token
     chunks, +2.0%), A19 (CUDA-only mma.sync +2.2-3%).
   - grim: **already has** WY chunk oracle + kernel: `crates/grim-backend-rocm/src/kernels/gla_kernels.rs:189-199`
     (`GRIM_GDN_WY_CHUNK`), `crates/grim-models/transformer/src/qwen4exp_gdn.rs:239-240,492`,
     `tests/gdn_wy_chunk_oracle.rs:6,67` (**defaults OFF**); `plan_journey_e2e.rs:180-236` covers production forward.
     No in-tree evidence of a matrix-core WMMA 16-token chunk or the MTP deferred-commit fix.
   - Transfer: (a) A/B `GRIM_GDN_WY_CHUNK=1` on real Flash-Next prefill first - cheapest test; (b) port RDNA3 WMMA
     prep/scan (Avalon `gdn-wmma.{cu,cuh}`) if WY path proves shape; (c) add deferred-commit replay-window for MTP.
   - Gate: `gdn_wy_chunk_oracle.rs` (exists) + `qwen4exp_gdn_wiring_test.rs` + Gate D.

3. **Prefill expert byte-ring + ubatch fitted to cache lend + compute buffers lent to expert cache** - `FN`, TTFT. **WIRE**.
   - Avalon: A27 (`5942fb7b`/`b7eacfc6`/`043ffd06`: byte ring, 4->8 guarded records, no host wait, default 900 MiB lent
     on HIP, ~10% fewer entries moved), A28 (`29e013b7`: largest 256-token-grid ubatch that fits cache lend, equal parts),
     A3 (`f5a34ece`: QSA slice 4096 => compute buffer 2750 -> 2191 MiB, less eviction), A7 (`7cd93b54`).
   - grim: `crates/grim-backend-rocm/src/device/expert_prefetch_stream.rs:3-83` + `GRIM_ASTRUM_EXPERT_PREFETCH` default OFF,
     instantiated only from unwired `moe_hybrid_exec.rs:160-178`; hybrid CPU/GPU overlap `moe_hybrid_exec.rs:16-105`
     consumed only by unwired `grim-nn/src/moe.rs:2060-2140`; Flash-Next forward uses host top-k + staged experts
     (`qwen4exp_flash_next.rs:774-834`); EPLB/route prior `device/eplb.rs:36-95` unwired.
   - Transfer: wire existing prefetch + hybrid overlap into `Qwen38MoeBlock::forward`, then add lend-aware ubatch sizing
     (env gate, default OFF until bench).
   - Gate: `moe_d2d_routing_gpu.rs`, token parity on real Flash-Next weights, A/B at 19k/131k prompt.

### Tier 2 - real but smaller, both models

4. **Backend greedy = bare argmax, no logits output, shared chain, two-stage argmax for 248k vocab** - `27B` (vocab 248320)
   and `FN`. **PORT**.
   - Avalon: A21 (`7fc48274`: skip top-k(40)+logits copy for greedy, 64 blocks/row then warp reduction, added kernels
     0.37 -> ~0.02 ms/step, 39.5 us/row at 248320 cols), A22 (`a3870717`: one shared chain removes 155-178 ms graph
     re-reservation), A23 (`1e193158`: sampler inputs in own buffer).
   - grim: device sampler `kernels/device_sampler.rs:283` caps at `1<<18` (fits 248320); device argmax
     `gpu_argmax_into_dev` `device/compute/kernel_infra.rs:215-241` (faults fixed at HEAD `9aa53b9a`), used by decode graph
     (`decode_graph.rs:5759,5915`); full-vocab CPU fallback `qwen35.rs:2689-2733`; **no frequent/cut-vocab**; padded vocab
     sliced `qwen35_mtp_adapter.rs:174-185`.
   - Transfer: greedy-only fast path that (a) drops logits as graph output, (b) two-stage argmax over 248320 cols on
     device, (c) shares one chain across requests. Smaller than Avalon's change - grim already has device argmax.
   - **Honesty caveat**: Avalon's `-bs` auto-enables only on CUDA; on RX 7900 XT it was neutral and left OFF
     (`common/common.cpp:1446-1459`, `README.md:236-237`); the 6%/step is the RTX 4090 number. Transfer target for grim
     is its own vocab-sized D2H cost, not an assumed ROCm gain - A/B before believing it.
   - Gate: argmax parity vs CPU over full vocab incl. ties; `cli_decode_parity.rs` token gate.

4b. **MTP-specific micro-transfers (mostly `FN`, small)** - **PARTIAL**.
   - Draft `eh_proj` restructured to one matmul over `hc × n_tokens` (2.9 ms / 256-token pass RX 7900 XT),
     KV-only draft passes (cache writes only, no Q/attention), a small-graph cache (+12%/step when pipeline path avoided),
     and draft joining target's expert cache as a "guest" pool (Appendix B2).
   - grim: MTP implemented default OFF (`GRIM_MTP`, `run.rs:1867-1869`), chained path fixed at HEAD but no win;
     no cut-vocab draft head, no KV-only draft passes.
   - Transfer: KV-only draft passes and single-matmul `eh_proj` are the only two with a clean parity story; FR-Spec
     cut head (65k of 248k logits) needs a requantized draft checkpoint - out of scope unless adopting Avalon's GGUF.
   - Gate: existing `scripts/check-mtp-semantic-gate.sh` control-pair gate; do NOT accept text diff as oracle
     (baseline generation nondeterministic at temp=0 with fixed seed).

5. **Context-checkpoint state into pinned host memory, deferred copy after first token** - TTFT for follow-up requests,
   `both`. **PORT**.
   - Avalon: A02 (`07dcefe8`: capture into pinned buffer, copy after first-token payload sent; server TTFT short
     42-43 -> 27 ms, 19k 45 -> 29-34, 131k 57 -> 40-43; byte-identical state on 14 checkpoints).
   - grim: radix/session prefix cache implemented (`grim-engine/src/scheduler.rs:216-350`, `lib.rs:272-289`), checkpoint
     snapshots CPU-pool (`grim-core/src/session.rs:192`); **no pinned deferred state restore** (only pinned D2H staging
     `storage.rs:263-343` for logits/histograms).
   - Transfer: capture recurrent + KV checkpoint into pinned host at prefill end, H2D overlapped with first-token sampling.
   - Gate: checkpoint byte-identity + TTFT A/B; reuses `tests/radix_prefix_reuse.rs:144`.

6. **Two-GPU pipeline: prompt pipelining + full-ubatch long prompts + hidden-state-only crossing** - `27B` (FN if 2 cards).
   **PARTIAL / FEEDS STAGE B**.
   - Avalon: A29 (`a40bb4f5`/`56581776`/`9e9cbb83`: layers split 23/25, per-stage KV/recurrent state/expert cache/thread,
     prompt pipelined across both PCIe links, batch raised to context size => 60k 1649 -> 1761 t/s, 120k 1451 -> 1598
     t/s, generation unchanged; `LLAMA_STAGES_FULL_UBATCH_FROM` for link-bound case).
   - grim: layer pipeline already the design (`plan_layer_devices` `qwen35.rs:7489`, called `:2995`; only hidden vector
     crosses). Stage A COMPLETE (`f063327a`); **REV 2 withdrew the <=45 ms target** (`plans/PLAN-tp2-decode.md:33-34`),
     the predeclared stop test then closed TP on failure (required <=80.03 ms, projected 72.9-85+ ms; pipeline split stays
     production default - see `project-tp2-plan-verification` memory); `GRIM_MOVE_P2P` OFF/broken
     (`grim-nn/src/modules.rs:404-415`), graph capture cannot span devices.
   - Transfer: Avalon's prompt pipelining and full-ubatch-for-long-prompt are concrete inputs to the **pipeline** path
     (not to TP); keep grim's hidden-state-only crossing (Avalon matches). Fix P2P before chasing rendezvous microseconds.
     Treat as a pipeline refinement, NOT a reopened TP case - the TP stop verdict is measured, not stale.
   - Gate: existing two-card A/B + ULP parity per `PLAN-tp2-decode.md`; no parity waiver for latency.

7. **Fused hyper-connection post + norm (image + inject written by producer)** - `FN` (4 residual streams / HC mixer).
   **PORT IF APPLICABLE**.
   - Avalon: A10 (`367cf8b1`/`3d5aa1fc`: 93/94 boundaries fused, 390 MB/call removed at -ub 9536, bit-identical hc_mixed +
     logits hash; +1.5% from A11), only -0.5% on RX 7900 because link-limited.
   - grim: no matching HC fusion anchor in the verified map - check `qwen4exp_flash_next.rs` hyper-connection path before
     claiming absence. Lower rank (link-bound on target RDNA3).
   - Gate: bit-identical hc_mixed + logits hash vs unfused, then Gate D.

8. **MoE tail skip (rank>=7 missed experts)** - `FN`, decode. **PARTIAL - LOSSY, gated only**.
   - Avalon: A24 (`510abcb1`: 91.6 -> 99.6 t/s MTP, 78.5 -> 88.5 after 19k; top-1 98.6-99.7%, KL 0.002-0.004).
   - grim: **absent** (no `tail.?skip` hits); but route-prior infra exists (`routing/moe_dispatch.rs:27-46`,
     `grim-memory/src/moe_budget.rs:415,491`) and a top-1+KL harness pattern (`tests/kv_quant_quality.rs`).
   - Transfer: opt-in env gate in grouped dispatch; gate with top-1/KL (not byte identity) + task eval before any default
     flip. Changes output - unlike items 1-7.

### Tier 3 - already covered, wrong backend, or not worth pulling

- **Raw quantized embedding rows / dequant on device** (A25): **NO** - grim already does this
  (`grim-backend-rocm/src/device/compute/core_tensor_ops.rs:698-734`, `fused_ops.rs:26-219`).
- **Expert cache + CPU/GPU overlap + route prior** (A30/A31): **NO as new algorithm; WIRE as implementation** - grim's
  P3 is PARTIAL (Appendix D1): CPU residency/prior primitives landed but the production route passes `None/0/0/0` and
  Flash-Next/27B device routing has no prior. P12's "prefetch stream" is a CPU stub with no HIP stream. Avalon adds
  production evidence for the wiring, not a new idea. Corrected 2026-11-20 after PLAN-astrum audit.
- **hipBLASLt/rocBLAS solution tables, GEMV rows-per-warp table, PDL, KVA skip, q4-hadamard KV, DraftPolicy, suffix
  drafter, conversation checkpoints**: **NO for algorithmic transfer** - Avalon converges on the same tactics, but
  "in PLAN-astrum" does NOT mean implemented: code census is P1/P2/P3/P4/P5/P7/P10/P12 = PARTIAL, P6 =
  PRE-EXISTS-UNWIRED, R-P9/P8/P11 = NO TRACES (Appendix D). Avalon's value here is a production-wiring/measurement
  template only; do not claim these are already covered by a shipped grim path.
- **k-group / activation sparsity skip**: **NO (closed)** - grim built and measured it; 27B production ceiling ~0.25%,
  default OFF, greedy degrades. Avalon's tail skip is a DIFFERENT (routing-rank) lever.
- **CUDA-only items**: GDN mma.sync TC2 (A19), dense Q8_0 cuBLAS F32 prefill, NVIDIA-default backend sampling,
  SYCL `top_k_radix_row` - **NO** for RDNA3 target; reference only.
- **IQ3_KS quant type** (Avalon model format): **NO for now** - absent in grim (`QuantFormat` has IQ3_S only); adopting
  Avalon GGUF needs new quant type + dequant + kernels + parity gates; large, model-specific, out of scope for grim's
  existing checkpoints.
- **answer_room** (A32): **NO for speed** - ms/token unchanged (10.754 vs 10.746). Relevant only to grim's known
  "full 2048-token think phase" wall; if adopted, as e2e/quality feature with own gate, not a speed lever.
- **MTP chain**: **NO** - grim built/fault-fixed it at HEAD; measured no win, opt-in/OFF. Avalon's 2-draft-token FR-Spec
  head is model-specific (needs requant/training).

### Suggested execution order (lazy path)

1. A/B `GRIM_GDN_WY_CHUNK=1` (zero code) -> port Avalon WMMA chunk only if WY path wins.
2. Wire existing grim QSA kernels into Flash-Next forward (stage 1, keep host oracle) -> port tile lists only if A/B wins.
3. Wire existing `expert_prefetch_stream` + hybrid overlap into `Qwen38MoeBlock::forward`; add lend-aware ubatch sizing.
4. Port bare-argmax backend greedy + shared chain (27B 248k-vocab D2H is the target).
5. Pinned checkpoint + deferred state copy (follow-up TTFT); feed the existing pipeline path (not the closed TP case)
   with the full-ubatch / prompt-pipelining idea.
6. Tail skip LAST, lossy gates first, default OFF.

Every step: change one lever, token-identity/parity gate first, then A/B with >=3 medians (bench noise +/-1 tok/s),
HIPRTC-appended kernels only.

## Appendix B: Avalon code anchors (verified by code-reading agent, 2026-11-20)

Paths below are workspace-relative to `/drive/bigfast/grim/old/repo/avalon/`.

### B1. QSA full stack (`FN` only; all under `src/models/qwen4exp.cpp`)
- Config/pooling: `:140-177` - `ATTENTION_COMPRESS_RATIOS` -> indexer key pooling in blocks; all QSA layers share one
  ratio; ratio >1 and divides `indexer_top_k`; semantics `indexer_kpool_row=2`, `by_order=true`, `select_tail=true`.
- Op/API: `ggml/include/ggml.h:590` (`GGML_OP_QSA_SELECT`), `:2724-2794` (modes `ROWS/MASK/DENSE/CAUSAL`,
  `POOL_MEMBERS/REP/END`, `TILE_ROWS/TILE_MASK`, API `ggml_qsa_select`, `ggml_qsa_select_tiles`, `ggml_qsa_pool_layout`).
- Selection: `build_qsa_sel :1134-1408` - pooled key = mean of block, norm+RoPE to first member; persistent k-pool cache
  `:1172,1194-1211`; short-context degeneration to causal `:1213-1226`; lightning indexer `:1258`; top-k `:1267`;
  expansion to member cells + incomplete tail `:1276-1286`; 512 MiB score/mask slices `:1321-1322,1351-1352`.
- Generation path: gather selected cells only (`GGML_QSA_SELECT_ROWS`+`MASK`) when
  `flash_attn && one seq && (fixed || 4*n_tokens*n_sel < n_kv)` `:1299-1317`.
- Prefill path: dense or tile mask `:1319-1407`; tile mask per 16 (or 8) queries `:1354-1399`.
- Graph-split reuse: `kpool_fixed` tables sized for whole cache, context-length independent;
  `src/llama-graph.h:3-6` `LLAMA_GRAPH_SMALL_MAX_TOKENS=8` explicitly covers speculative verification windows and draft.
- Auto-growing pool layout preserves pools before an edited position (speculative reject removes draft tokens without
  full rebuild): `src/llama-memory-hybrid-idx.cpp:473-542` (comment `:514-518`).
- Attention over gathered cells: `build_attn_qsa :1410-1545`; q8_0 K / q4_0 V stay quantized for sparse gather
  `:1481-1498,1505` (`kv_direct ... k->ne[2] >= 65536`); sliced dense path `:1500-1529`;
  `build_layer_attn_qsa_sliced :1679-1716+`.
- GPU kernels: `ggml/src/ggml-cuda/qsa-select.cu:10-43` (`k_qsa_select`), `:45-75` (`k_qsa_pool_layout`),
  `:77-202` (`k_qsa_tile_rows`: shared-memory bitmap, 512 threads, max 480k cells = 60 KiB, warp-scanned sorted list;
  `k_qsa_tile_mask`), `:262-344` dispatch; `qsa-select-body.cuh:8-52`.
- Fused top-k + QSA select: `ggml/src/ggml-cuda/top-k.cu:301-365` `top_k_radix_fused` - 11 launches of ~1.5 us
  become one (HIP-only, `top-k.cu:910-934`).
- Sparse flash attention: `ggml/src/ggml-cuda/fattn.cu:9-115` (`flash_attn_mask_to_sparse_indices`, RDNA `__ballot`),
  `:137-203` (`shall_use_sparse`/`kv_quant_sparse`, tile mask only in `(256,256,16,4)` MMA kernel), `:826-829`
  (no f16 K/V copy when quantized sparse); unsupported on MUSA `:117`.
- Env flags: `LLAMA_QSA_TILE_COLS` (8/16, default 16) `qwen4exp.cpp:1110-1115`; `LLAMA_QSA_TILES` /
  `LLAMA_QSA_TILES_ALL` default on `:1357-1360`; `LLAMA_QSA_ATT_SLICE` default 4096 `:1361-1384`;
  `LLAMA_QSA_FIXED` `llama-memory-hybrid-idx.cpp:1250`; `LLAMA_QSA_DEVICE_LAYOUT` `:1263`;
  diagnostics `GGML_CUDA_QSA_TILE_STATS` `fattn.cu:856-884`, `qsa-select.cu:204-258`.

### B2. MTP (`FN`, FR-Spec head; `27B` relevance limited)
- Cut LM head via `d2t` (frequency-ranked 65k tokens): `qwen4exp.cpp:275-300`; scatter cut logits into full vocab with
  `ggml_fill(-INFINITY)` + `ggml_set_rows` in `graph_mtp :889-903`. FR-Spec list in GGUF filename `README.md:229`.
- Draft graph: `graph_mtp :766-798` (`embd_h`, `h_src`, on-device `mtp_h_input_dev`, embedding gather first to keep
  single graph split); one QSA MTP block `:852`; next-step hidden from `t_h_nextn :869-874`;
  `LLAMA_MTP_KV_ONLY=0` disables KV-only prompt passes `:810-813`.
- Driver `common/speculative.cpp:1391-2133`: modes `:1405-1411`; CPU `top_k=10` sampler `:1475-1482`;
  per-ubatch device row callback `:1508-1512,1580-1650`; `pending_h`/`verify_h` carryover `:1413-1432`;
  `draft :1897-2111`; greedy keeps top candidate `:2039`; p_min drop `:2041-2047`; `accept :2113-2132`.
- Draft shares target's GPU expert cache as "guest" pool: `ggml/src/ggml-cuda/moe-cache.cu:329,622,680-681,4088`,
  host memory sharing `:3944-3983`, ~127 MiB draft compute buffer kept preferentially `:2755-2757`.
- `eh_proj` in prompt restructured to one matmul over `hc × n_tokens`: 2.9 ms / 256-token pass on RX 7900 XT
  (`qwen4exp.cpp:834-841`). KV-only draft passes `:847-850,1567-1597`.
- Small-graph cache avoids pipeline path for draft: `src/llama-context.cpp:610-611` (+12%/step when it hits).

### B3. Sampling honesty note
- Avalon's `-bs` auto-enables ONLY on CUDA (`common/common.cpp:1446-1459`); on RX 7900 XT it was **neutral** and left
  OFF (`README.md:236-237`). Claimed 6%/step is the RTX 4090 number. The transferable part for grim is the
  bare-argmax / no-logits-output / shared-chain structure, NOT an assumed ROCm speedup.
- `ggml/src/ggml-cuda/argmax.cu:70-73,82-169`: `ARGMAX_SPLIT 64`, split when `ne00>=32768 && nrows<=65535`,
  ties to smaller column; 39.5 us for 248320 cols on RDNA3.

### B4. Fusion infrastructure
- Multi-task kernel `mk.cu:35-44,81,202-211,288-294,404-405,474-481,523-648`: fuses ROPE, FWHT rotation, FILL,
  QSA_SELECT, K/V gathers; `ggml-cuda.cu:7118-7201` classifies nodes.
- Graph fusion pass `ggml_cuda_try_fuse` `ggml-cuda.cu:5308-5450`: pooled-mean `:5316`, gather+cast `:5333`,
  FWHT-gate `:5440`, FWHT set_rows `:5442`, qwen4 HC `:5450`; invoked at `:7358`.
- Graph-split-count discipline: `qwen4exp.cpp:1433-1446` queues K/V stores together to reduce graph splits.
- Caveat: the "53 -> 39 launches" figure is a README claim (`README.md:182`), not a code literal.

### B5. Pipeline
- `LLAMA_STAGES` default ON with 2 GPUs; `LLAMA_STAGES_FULL_UBATCH_FROM`, `GGML_MOE_CACHE_LAYER_DEVICE=1`
  (`README.md:253-283`); default 23/25 split with "+27 to +45% prefill over layer split, generation equal"
  (`src/llama-context.cpp:177-180`); per-stage thread/memory `:95,197,2804,2889`; hidden-state rows cross `:95,3117-3187`;
  MTP rows per ubatch on device with events `:2773-2776`; batch raised to context => one call (`README.md:259`).
- Limitation measured only on NVIDIA pairs; untested on two AMD GPUs (`README.md:263-283,289-290`).

## Review completeness audit (self-check)

| Requirement | Evidence |
|---|---|
| Repo cloned to `old/repo/avalon` | `git log -1` = `26a2155`; 228 MB checkout (this session) |
| Review against grim | grim HEAD `9aa53b9a`; clean tree; both sides code-read |
| TTFT covered | Items 1-6 + A02/A13/A15/A17/A18/A29; prefill/TTFT section of both agents |
| Decode speed covered | Items 4/8 + A21/A22/A24/A31; MTP backend-sampling section |
| Qwen3.8-27B addressed | Items 4-6, 8; 27B-specific data from grim memory (248320 vocab, 30 tok/s chain) |
| Qwen3.8-Flash-Next addressed | Items 1-3, 7, 8; `qwen4exp` model/quant/pipeline anchors |
| Thorough / meticulous | 500-commit Avalon history mined, 32-claim evidence appendix, 6 code-reading agents delivered (Appendices B, C, D: 3 Avalon + 3 PLAN-astrum), code anchors on both sides, PLAN-astrum audited phase-by-phase into four buckets, duplicate/closed levers explicitly reclassified, README-only claims flagged and two inert flags corrected |
| Skills used | `ai-inference-expert`, `ai-llm-inference`, `caveman-mode`, `ponytail` invoked |
| No phantom claims | Every item has Avalon commit or code path + grim path; README-only claims marked |

All three code-reading agents delivered. No transfer recommendation depends solely on a commit-message claim;
Tier-1/2 items are anchored in appendix A commits plus the delivered code reports. A material correction from the
MoE agent is recorded in Appendix C.

## Appendix C: Avalon MoE code anchors (verified) + correction

Paths relative to `/drive/bigfast/grim/old/repo/avalon/`. Provenance from code comments: cache core = leloch
(RFC ggml-org/llama.cpp#24528) `ggml/src/ggml-cuda/moe-cache.cu:1-3`; buun = GPU cache + LFU + ranking + RAM tier
(`README.md:302-304`); thecodacus = prefetch (`README.md:305`); ik_llama.cpp = IQ3_K/KS (`README.md:306`);
Strata = ranking format + tail skip (`README.md:308`).

### C1. GPU expert cache (live)
- Entry = whole expert (up/gate/down, 256-byte-aligned): `moe-cache.cu:152-160,223-228`; key = (gate tensor ptr, expert id)
  `:162-181`; LFU `freq` `:186-194`; pools per shape `:309-318`.
- Calculated reserve: `moe_cache_recommended_reserve_bytes()` `:1167-1176` = `min(400 MiB, total_vram/4)`; measured peak
  ~170 MiB; NVIDIA floor 1024 MiB `:819-828`; override `GGML_CUDA_MOE_CACHE_RESERVE_MB` `:1077-1080`.
- Sized from unused VRAM via sysfs (hipMemGetInfo under-reports ~1.8 GiB on 20 GiB): `moe_cache_sysfs_free() :2466-2494`,
  `moe_cache_prepare_budget() :2552-2631`; multi-GPU split `:830-993`.
- Compute buffers lent between phases: `moe_cache_borrow() :7163-7328`, `moe_cache_plan_borrows() :7065-7159`,
  `give_back() :7345-7397`, `restore() :7400-7437`, 95% cap `:466`; bands `:2737-3019`.
- KV tail lent (default on): `GGML_CUDA_MOE_CACHE_KV_LEND` `:5870-5883`, attach `:5886-5898,6353-6411`.
- LFU: `moe_cache_lookup_or_queue_locked() :5205-5248` (admit_after 3 / readmit 8 / cap 15), aging `:5278-5295`.
- Ranking/preload (STRP v1/v2 + GGMLMHC1 heat): `moe_cache_ranking_read() :5938-5963`, `moe_cache_preload() :6053-6338`;
  auto ranking file `common/common.cpp:1316-1326`; profiling `moe-cache.cu:5965-6051`.

### C2. Hosted MoE and expert split (partly live - see correction)
- Graph-capturable mailbox protocol: `moe-cache.cu:7439-7451`, capture hooks `:10872-10944`, wait `:10228-10347`,
  block `:11386-11730`; `moe_hosted_plan() :8912-9102` (12.5 us / 30 rows), `moe_hosted_serve_one() :9909-10165`.
- Expert split (move 1-2 misses to GPU): `moe-cache.cu:9104-9139`, `moe_hosted_split() :9174-9558`; cost model fitted
  on ~15k 3-token MTP layers; PCIe 3.0 ~13.6 GB/s; `moe_hosted_cost() :10575-10603` (4900 us fixed + 4.9 us/row,
  crossover ~12k rows on 7900 XT, ~5k on 4090).
- AVX2 CPU expert kernels: `ggml/src/ggml-cpu/ggml-cpu.c:4566-4570,4615-4789` (IQ2_S/IQ3_K/IQ3_KS/IQ3_XXS/IQ3_S/
  IQ4_XS gate/up; IQ4_NL/Q2_0 down; q8_0 GPU activations byte-identical; L3-group scheduling; <=256 rows).
- **CORRECTION**: `--moe-cache-cpu-overlap` is **parsed but inert in this tree** (`common/arg.cpp:2794-2811`,
  `common/common.h:480` default -2, plumbed through `common.cpp:1872`/`llama.h:411`/`llama-context.cpp:1004`/
  `ggml-backend.cpp:3576`), but no consumer reads `config.overlap_cpu_rows` behaviorally; the plan/dispatch/fused_begin
  consumers are `nullptr` (`moe-cache.cu:11784-11788`). README:137's "2,024 -> 2,113 t/s with `--moe-cache-cpu-overlap 2`"
  is a USER REPORT on a different binary and is NOT live code here. Correct A31 accordingly: the GPU-resident + CPU-miss
  hosted protocol exists, but do not cite the flag's overlap as a working feature.

### C3. Prefetch ring and RAM arena (live)
- 4 prefetch slots default: `ggml/src/ggml-backend.cpp:3509-3522` (`GGML_SCHED_PREFETCH_EXPERTS`; measured 19k prefill
  3 slots 1,353 -> 4 slots 1,446 t/s, 5-6 unchanged), slots borrowed from cache `:2530-2606`, resident read in place
  `:2890-2926`; provider hooks `moe-cache.cu:6596-6815`, API `ggml-backend-moe-cache.h:145-175,208-222`.
- Ring variants: `GGML_SCHED_PREFETCH_RING_MB` `ggml-backend.cpp:2326-2341`; `GGML_CUDA_MOE_RING_PREFILL`
  `moe-cache.cu:4212-4229,1525-1528,2690-2711`; `ggml-cuda/moe-ring.h:1-18` (7 IQ3_KS shapes, ~22% zone loss with
  uniform entry size - the byte-sized-entry fix is `043ffd06`).
- RAM arena: design `ggml-cuda/moe-arena.h:1-20,32-61,86-261` ("VRAM first" rule `:11-13`);
  `GGML_CUDA_MOE_ARENA_MB` `moe-cache.cu:3361-3378` (auto on Linux / 0 elsewhere), auto = room - 4096 MiB margin
  `:3395-3432` (MemAvailable + cgroup v2); O_DIRECT into owned anonymous memory + in-place GPU registration
  `moe_cache_pin_load_direct() :3147-3254`, `moe_cache_pin_experts() :3255-3360`; only arena-held experts admitted
  `:4837-4854`; multi-GPU RAM budget `moe-ram-budget.h:1-108` (10 GiB + 3 GiB/card reserve).

### C4. Tail skip and IQ3_KS (live)
- `GGML_CUDA_MOE_TAIL_SKIP` default **7** (1-9, 0 off, output-changing, main model only): `moe-cache.cu:9852-9862`;
  `moe_hosted_tail_skip() :9857-9907` - a missed expert skipped iff never routed earlier AND all its rows have slot
  `q % n_used >= 7`; skipped rows zeroed, NO renormalisation; counters `:9899-9905`; called at `:9954` before split.
- IQ3_K = enum 43, IQ3_KS = enum 44 (block scale, 104 B/256): `ggml/include/ggml.h:433-434`, `ggml.c:837-851,8361-8362`;
  GPU MMV `moe-cache.cu:9623-9626,11584-11620`, `mmvq.cu:2595-2685`; whole-expert packed entries with pad for MMQ
  overreads `moe-cache.cu:152-160,223-228,2371,6286-6293`; layer names `blk.N.ffn_{gate,up,down}_exps :320-330`.

### C5. What Appendix C changes in the ranked findings
- Item 3 (expert ring / prefetch / lend-aware ubatch): **strengthened** - all four mechanisms are live in Avalon and
  absent/unwired in grim; this is the most directly transferable MoE TTFT package.
- Item 8 (tail skip): **strengthened** - exact rule (rank >= 7 on EVERY row, never routed earlier, zero rows, no
  renormalisation) is now specified; grim can copy the rule and gate it with top-1/KL.
- A31 table row / Tier-3 "expert cache + CPU/GPU overlap" bullet: **corrected** - hosted protocol and expert split are
  live, but the `--moe-cache-cpu-overlap` knob is inert; do not transfer the README's overlap number. Route prior remains
  a PARTIAL, non-production-reaching P3 item (Appendix D1) - duplicate as an idea, not as working code.
- IQ3_KS remains a NO for now (new quant type, model-specific), now with exact enum/layout evidence.

## Appendix D: PLAN-astrum code audit (2026-11-20)

Scope: `/drive/bigfast/grim/plans/PLAN-astrum.md` (648 lines, 12 phases) audited against grim HEAD `9aa53b9a`,
clean tree, by three independent code-reading agents. Buckets, exactly as requested:
**FULLY DEVELOPED** (implementation + wiring + gate present and production-reachable),
**PRE-EXISTS, NO WIRING/PRODUCTION CALLERS** (assets complete, no production caller),
**PARTIAL WORK** (some steps landed; rest missing/failed/deferred),
**NO TRACES FOUND** (no implementation; needs implementation from scratch).

**Headline: zero phases are FULLY DEVELOPED.** Distribution: 0 fully developed / 1 pre-exists-unwired / 8 partial /
3 no-traces. The plan's "Phase results" section (`:642-648`) is stale or silent on nearly all of this: P2, P10, P7 and
P12 have landed code but no results entry; P4's results entry claims "no product source changed" while commit
`21c6ce60` landed product code; P7 is labelled STOPPED but its planner landed.

### D1. Per-phase status table

| Phase | Bucket | Landed (path:line, commit) | Missing / blocker |
|---|---|---|---|
| P1 BLASLt/rocBLAS tables | **PARTIAL** | `device/gemm_tuning_measured.rs:22-185` (MeasuredTable, loader, lookup, version guards, OnceLock), `gemm_tuning.rs:144-147` override, production `gemm_launchers.rs:67,112,421,514,574,2858`; `tests/blaslt_solution_table.rs` 2 CPU tests (no ignore); `tests/tune_blaslt_qwen35.rs:30` (`#[ignore]`, GPU) + FFI `blaslt.rs:19,28,267,363,468`; `6357aac3`, `847a85fb` | GPU measurement inconclusive (all hipblasLt matmul status 6 on gfx1200, `entries=[]`); no table JSON in `data/`; version guard `table_version_matches_probe` dead (no caller); `lookup_measured_blaslt_solution` has zero production callers (only rocblas wired); no default table path; no PPL sanity |
| P2 GEMV dispatch table | **PARTIAL** | `device/compute/gemv_dispatch_table.rs:15-145` (GemvVariant 5 variants, GemvKey, lookup_variant, current_variant), **production wiring** `dot_gemv.rs:750-754` (`current_variant` first, Plain falls through), `tests/gemv_dispatch_parity.rs:7` 1 CPU test; `6012f031` (+ `a4466cf8`) | `src/bin/tune_gemv.rs` missing; no generated table; Step 5 GPU gate never run; no Phase-results entry. Env `GRIM_ASTRUM_GEMV_TABLE` unset = OFF |
| P3 MoE residency + route prior | **PARTIAL** | `grim-memory/src/moe_budget.rs:332,371,453,415,491` (ExpertRouteTrace, ResidentProfileFreq, route_bonus), `tests/expert_residency.rs`; `moe_hybrid_exec.rs:105-220,275,317` (config/state, plan_step_with_astrum), `routing/moe_dispatch.rs:30-111` prior helpers, `lib.rs:64`; `grim-nn/src/moe.rs:2071-2157`; `shared_moe.rs:2738-2786`; `3a67515c`, `d276e595`, `d33a2daa` | Planned `expert_residency.rs`, `kernels/route_prior.rs`, `tests/expert_residency_parity.rs` absent. **No production effect:** `route_topk` (`shared_moe.rs:2731`) always passes `None/0/0/0`; astrum hybrid variant's only direct caller is a test (`moe.rs:4123`); `AstrumMoeResidencyState::from_env()` has no caller; Flash-Next/27B device routing (`qwen4exp_flash_next.rs:2965`, `decode_graph.rs:2070`) has no prior. No GPU/PPL/perf gates. λ default 0.0 (plan wanted 2.0); trace writes in-memory only, not `<outfile>` JSON |
| P4 Q4_0 KV + Hadamard | **PARTIAL** | `grim-kvquant/src/lib.rs:278` named_profile `q4_hadamard` (+`:299` NAMED_PROFILES), `:119,124-132` `KvRotationMode::Hadamard256` gated on `KV_ROTATION`/`GRIM_KV_QUANT`, `:137,158` `fwht_inplace`/`hadamard_matrix`, rotation at `:839-849`, inverse at `:1050-1060`; `tests/q4_hadamard.rs` 4 CPU tests; engine `core.rs:45,68`; `21c6ce60` | `hadamard256()` symbol absent (equivalent `hadamard_matrix(dim)`); test file renamed (planned `q4_hadamard_profile.rs` absent); GPU parity test missing (attempted, max_err ~0.622 vs 2e-2, removed); PPL gate not run. Plan results entry contradicts landed product source |
| P5 online draft policy | **PARTIAL** | `grim-speculative/src/draft_policy.rs:1-155` (DraftSource, DraftPolicy, EMA select/observe, `from_env`), `lib.rs:16,28`, `speculative_wrapper.rs:99-102,149,195,218,692-700`; `tests/draft_policy.rs` 5 CPU tests; `9f9de7e5`, `ff133a1d`, `6eae4d89` | **Inert:** `:694-699` hardcodes `DraftSource::Mtp` + `cost_ms=1.0`; `select()` result discarded (`let _ =`); no Suffix observation in production; `try_draft_astrum_suffix` zero callers. Plan's engine file `grim-engine/src/speculative.rs` call site absent; `record_round_start/end` absent; margin default 0.05 (plan 0.10); no GPU bench/output-neutrality gate |
| P6 trigram suffix drafter | **PRE-EXISTS, NO WIRING/PRODUCTION CALLERS** | Complete module `suffix_drafter.rs:1-181` (new/append/draft, ring + trigram index, `MAX_POS_PER_TRIGRAM=4`), `lib.rs:18,39`, wrapper instance + method `speculative_wrapper.rs:102,150,196,219,296-319`, `tests/suffix_drafter.rs` 5 CPU tests; `3ecc5684` | `try_draft_astrum_suffix` has **zero callers**; no env knob; no engine admission of suffix drafts; no GPU gate |
| P7 KVA late-layer prefill skip | **PARTIAL** | `grim-engine/src/kva_prefill.rs:15-73` (`LayerPrefillMode`, `KvaSplitPlan::plan` with tail>=1, `from_env_str`, `from_env`, `split_prompt`), `lib.rs:3-5`, `tests/kva_prefill.rs` 2 CPU tests + `plan_journey_e2e.rs:180`; `2b9032d5` | Planner landed despite plan's STOPPED label. **No production consumer:** `from_env`/`split_prompt` only in module + tests; prefill loop unmodified. **Precondition still fails:** no per-layer state-only entry point (gdn2_step `gla.rs:133` computes output; `qk_rope_append_kv_into` `rope.rs:461` KV-append only; `launch_gla_state_update_output*` `gla_launchers.rs:49,144` produces output). PPL ≤0.01 and ≥1.10× gates absent |
| P10 conversation checkpoints | **PARTIAL** | `grim-memory/src/conversation_checkpoint.rs:12-103` (ConversationCheckpoint, RestoredState, Cache cap-4 LRU, gated env), `src/lib.rs:55,220,305-306,3429,3442` gated save/restore methods, `tests/conversation_checkpoint.rs` 5 CPU tests; `f20a9d60` | **No production caller** (only definitions + `plan_journey_e2e.rs:370-373`); engine recurrent flow uses `scheduler.rs:297,426,441` instead, unconnected. No hybrid two-turn gate (turn-2 prefill ≤50%, greedy identity) |
| P12 PCIe expert prefetch | **PARTIAL** | `device/expert_prefetch_stream.rs:11-81` (env gate, PrefetchLayerTicket, ExpertPrefetchStream, enqueue/sync), `device/mod.rs:16-17`, constructed in `moe_hybrid_exec.rs:156,167-171`, `tests/expert_prefetch_parity.rs:7` 1 CPU 10-pass test | `enqueue_next_layer_prefetch`/`sync_layer_ready` called only by the test; MoE layer dispatch never issues them. **No real HIP stream:** staging is host `Vec<u8>` (`:48,55-58`), `sync_layer_ready` just sets a bool (`:76`), zero hipStreamCreate/hipMemcpyAsync — module doc's "secondary HIP stream" is untrue. No GPU bit-exact stress gate, no bench |
| R-P9 prefill FA2 | **NO TRACES FOUND** (deferred by plan) | No `GRIM_ASTRUM_FA2_PREFILL`, `fa2`, or fused-prefill feature. Cited `extend_attention.rs`/`prefill_compact.rs`/`flash_decode.rs` pre-exist but are not R-P9 deliverables | Needs implementation from scratch if revived; Gate D + oracle + bench required |
| P8 LDS expert kernel | **NO TRACES FOUND** (removed) | No LDS grouped-expert-decode kernel / env symbol | Out of scope |
| P11 PDL | **NO TRACES FOUND** (removed) | No `ProgrammaticStreamSerialization` / PDL / env symbol | Out of scope |

### D2. What this changes in the Avalon transfer verdicts

- **Item 3 (expert ring / prefetch / lend-aware ubatch) - strengthened.** P12 is PARTIAL with a *fake* CPU-only
  prefetch stream (no HIP stream, no production enqueue). Avalon's `moe-ring.h` / `GGML_CUDA_MOE_RING_PREFILL` is a
  real, production-default-on HIP implementation. Verdict stays WIRE, but note it requires **replacing** grim's P12
  stub, not merely calling it. P3 resident/prior is also PARTIAL and cannot affect current decode routing.
- **Item 4b (MTP micro-transfers) - unchanged.** P5 is PARTIAL-inert (hardcoded MTP, suffix never observed), P6 is
  PRE-EXISTS-UNWIRED. Both are code-shaped but non-functional here; Avalon's value is the production wiring pattern,
  not the algorithms.
- **Item 5 (pinned checkpoint) - refined.** P10 is PARTIAL: a message-boundary checkpoint cache exists in
  `grim-memory` but has no engine caller and does not do pinned deferred H2D. Transfer remains PORT; reuse P10's
  data structure, add Avalon's pinned-capture + after-first-token copy, then wire the engine call site.
- **Tier-3 "duplicates" bullets - corrected.** The earlier blanket "already in `PLAN-astrum.md`" claim was too strong:
  a PLAN is not an implementation. Precise status: P1/P2/P3/P4/P5/P7/P10/P12 = PARTIAL, P6 = PRE-EXISTS-UNWIRED,
  R-P9/P8/P11 = NO TRACES. None has a passing end-to-end GPU gate. Therefore Avalon is still useful as a **wiring
  and measurement template** for those phases; it is NOT a source of wholly new algorithms for P1/P2/P4/P5/P6.
- **No new items promoted.** Nothing in the Astrum audit adds a transfer target beyond items 1-8; it changes their
  effort estimates (more existing scaffolding, but no finished path) and removes the excuse "it's in the plan".

### D3. Plan-vs-code contradictions worth recording

1. Plan results `:642-648` omit P2, P10, P7, P12 entirely despite landed code with commits (`6012f031`, `f20a9d60`,
   `2b9032d5`, plus P12's stream module).
2. P4 result says "experimental test removed; no HIP/kernel/product source changed" - false: `21c6ce60` landed
   product profile + FWHT-256 rotation + compressor wiring + CPU tests.
3. P7 header says "STOPPED (precondition missing)" - planner code exists and is exported; blocker (no state-only
   per-layer entry points) remains real, so STOP verdict stands for *execution*, but "not implemented" is stale.
4. P1 result claims "version guards landed" - `table_version_matches_probe()` has zero callers in production;
   the blaslt override has no production caller (only rocblas path is reachable).
5. P5 margin default differs (code 0.05 vs plan 0.10); P3 prior λ default differs (code 0.0 vs plan 2.0).
6. P12 module documents a "secondary HIP stream" and "completion event"; code has neither (host Vec + bool flag).
   Do not trust module docs here - read the implementation.

Standing rule reinforced: plan checkboxes and "Phase results" are not evidence. The four-bucket code census is.
