# Avalon -> grim transfer review: TTFT + decode speed for Qwen3.8-27B / Qwen3.8-Flash-Next

Review date: 2026-11-20. Reviewer: ZCode. Status: COMPLETE (14 code-verification agents delivered: 3 Avalon, 3 PLAN-astrum, 3 Basalt, 3 llama.cpp 1.0, 2 dense-specific; see Appendices B-G).

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
| Thorough / meticulous | 500-commit histories mined for Avalon, Basalt and llama.cpp 1.0; 32-claim evidence appendix; 14 code-reading agents delivered (Appendices B-G: 3 Avalon + 3 PLAN-astrum + 3 Basalt + 3 llama.cpp 1.0 + 2 dense-specific); PLAN-astrum audited into four buckets; Avalon/Basalt/llama1 concepts split NEW/PARTIAL/DUP/N/A; explicit dense/27B coverage (F1 llama1, G1 Avalon, G2 Basalt); duplicate/closed levers reclassified; README/commit-message-only numbers flagged; two inert flags corrected |
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

## Appendix E: Basalt transfer concepts (2026-11-20)

Scope: `https://github.com/jesdga95/basalt` cloned to `/drive/bigfast/grim/old/repo/basalt`, HEAD `a365240`
(24 MB, C++/CUDA, Strata fork). Basalt targets **exactly** Qwen3.8-27B (dense, DFlash2) and Qwen3.8-Flash-Next
(512-expert MoE, MTP/DFly2), plus DeepSeek-V4.1-Flash and Hy4-flash, on NVIDIA Blackwell sm_120. 500 commits mined;
three code-reading agents verified the mechanisms. **Platform boundary: all CUDA kernels sm_120-specific. Only the
architecture-neutral algorithms below are transferable to grim's Rust/HIP RDNA path.** All paths relative to
`/drive/bigfast/grim/old/repo/basalt/`.

Status legend: **NEW** = no equivalent in grim; **PARTIAL** = grim has an inert/partial analog; **DUP** = already
covered (Avalon/strata/PLAN-astrum); **N/A** = model-specific, not applicable to grim's two targets.

### E1. Worthwhile concepts, ranked

1. **`DraftPolicy::may_verify` - stop drafting before a head pass no window can verify** - `both`, decode. **NEW, highest value.**
   - Code: `include/basalt/spec/draft_policy.hpp:51-61`, `src/spec/draft_policy.cpp:195-214`, proof test
     `tests/spec/draft_policy_test.cpp:154-205`. Rule: with chances for drafts `0..j` known, compute best
     `E(t)/cost_ms(t)`; for unknown later drafts assume `best_rate()` (cpp:64-71); if no completion can beat the known
     best (probe exception `kProbeChance=0.85`), return false. Engine passes `need` to `draft_end`
     (`program/serve/session_engine.cpp:533,814`); DFly2 checks confidence BEFORE the full-vocab head
     (`models/hy4flash/dfly.cpp:459-488`). Output unchanged by construction (only drafts not made). Defaults:
     `margin=0.03`, `kMaxT=16`, chance bins `{0.3..0.995}`, `kChanceDecay=0.98` (hpp:35-40).
   - grim: P5 draft policy is PARTIAL-inert (`grim-speculative/src/draft_policy.rs:1-155`, wrapper
     `speculative_wrapper.rs:692-700` hardcodes `Mtp`/`cost=1.0`, `select()` discarded — Appendix D1). No `may_verify`
     anywhere. Pure C++/`<algorithm>` → direct Rust port; needs per-chance EMA + per-window cost + a `need` callback
     before the draft head.
   - Gate: soundness test (a ruled-out draft appears in no later window) + token identity; no GPU required.
   - Honesty: soundness is proven by the test; the standalone speed figure is qualitative in docs, not stored numerically.

2. **Use-capped expert victim selection (count, halve, MRU tie-break)** - `FN`, decode hit rate. **NEW.**
   - Code: `include/basalt/core/tier_policy.hpp:35-46` (`kLookCap=15`, `count_look`, `yields_before`,
     `use_half_life`), hooks `src/core/expert_tier.cpp:781-789` (`begin_token` halves every `victim_half_life`,
     `forget_use` at prompt start), `:814` count lookups incl. hits, `:893-924` `oldest_slot`, `:954-994` `claim`.
     Hy4 defaults `kVictimHalfLife=256`, `kAdmitUses=2`, `kClaimsPerRow=128` plus per-layer slot weights
     (`models/hy4flash/model.cpp:25-31,40-43`).
   - grim: no use-based expert eviction — `moe_budget.rs:267` LRU slot allocation, `expert_prefetch_stream.rs`
     CPU stub, P3 prior unwired. Pure `uint16` counters + timestamps, no CUDA.
   - Gate: frozen-tier token identity + hit-rate A/B (do NOT trust the "10-15% fewer misses" figure — agent found it
     only in the commit message, not in code/docs).

3. **Bounded expert-tier lend with MRU hot-slot compaction + double-buffered ring** - `FN`, TTFT + next-turn decode. **NEW (concept).**
   - Code: `expert_tier.hpp:82-84,159-165,356-368`; `expert_tier.cpp:1245-1278` segmented arena (256 MiB segments,
     depth-interleaved slots), `:1298-1347` `keep_hottest` (MRU-first D2D into kept prefix), `:1349-1387`
     `lend(bytes)`, `:1392-1433` `reclaim`; prompt use `models/hy4flash/session.cpp:406-433`; ring
     `session.cpp:1210-1329` (2 buffers, 2 threads/events, `ring_ahead` starts next layer's copy early).
   - grim: no `lend` semantics in backend/engine (grep-confirmed), no MRU expert compaction; Avalon A27 lends
     compute buffers and KV tail but does not preserve hot experts across the loan. Concept (hot-slot compaction,
     tail-segment truncation, device-side metadata) transfers; VMM/D2D mechanics must be reimplemented.
   - Measured in commit `92f72a9f` (message-only): 3.5K prompt lend 6.13 GiB keeping 4045/5699 slots,
     2528→2134 ms; next turn 480→379 ms; 32 decode tokens 928→789 ms (hits 58→69.7%). Numbers from commit text,
     not a stored doc; treat as directional.

4. **Device-side sliding-window metadata → one graph per layer** - `FN`/any SWA model, decode. **NEW, high, cheap.**
   - Code: `models/hy4flash/kernels/ops.cu:270-284` `swa_meta_kernel` computes row spans (first/count/position)
     on device from one pinned `pos0` word; call `session.cpp:1016-1017`, graph branch `:1878-1891`. No host meta
     writes inside the captured region → one graph replayed for any position instead of three graph/launch boundaries.
   - grim: graph capture ON by default (`decode_graph_buffers.rs:3847-3858`) but position/meta writes remain host-side
     in several paths; per-M pre-capture exists (`decode_graph.rs:5923-6027`). Move window-span derivation to a device
     kernel reading a device word; keep the same accumulation order.
   - Gate: token identity under graph replay + `GRIM_KERNEL_TIMES=1` launch-count A/B.

5. **Fused add + RMSNorm + hyper-connection post with byte-identical reduction order** - `FN`, decode/prefill. **NEW pattern, caution.**
   - Code: `models/hy4flash/kernels/ops.cu:211-268` `add_norm_hc_post_kernel<3072>`; parity `kernels/kernels_parity.cpp:158-231`
     (`memcmp` bitwise vs separate kernels, 1/2/8/12 rows, single-among-12). Call sites `session.cpp:1885,1905,1919`.
   - grim: related fusion was REJECTED (`79b865a4` session memory) because combining recurrent norm+quant changed bytes
     and greedy text diverged. Basalt succeeds precisely by preserving the exact `__shfl_xor` + fixed `red[16]` sum
     order. Lesson: fusion is allowed only when the reduction order is reproducible; add a byte-identity gate first.
   - Note: agent could NOT find the literal "2x41 launches fewer" / "2290→1778 us" numbers in repo; mechanism and
     runtime-printed parity harness exist, numbers are commit-message claims only.

6. **Share one cell-tile load/conversion across multiple query heads** - `FN` QSA, prefill. **NEW idea (CUDA MMA impl not portable).**
   - Code: `models/hy4flash/kernels/mla.cu:440-444,487-493,660-699` (`mma_groups()==2` when `heads%32==0`,
     grid `dim3(heads/32,n_tok)`, two 16-head groups share each 32-cell tile load + fp16 conversion; fixed sum order
     keeps bits). Parity harness `kernels_parity.cpp:1053-1083`.
   - grim: QSA production path is host-materialized K/V + host mask (`qwen4exp_flash_next.rs:1504-1710` — item 1);
     once wire-less, this is the natural attention-side follow-up. Compile-time group param + forced fallback.

7. **Quantize a normed activation row once, reuse across sibling projections** - `both` drafter, decode. **NEW (byte-identical variant).**
   - Code: `models/hy4flash/dfly.cpp:258-272` (`mm()` quantize+MMVQ vs `mmq()` on pre-quantized rows, shared `in_q8`),
     used at `:386-390` (q_a/kv_a/gate), `:412-414` (MLP gate/up), `:472-476` (HC gate/up). HEAD commit `a365240`
     message: "15 launches fewer a block pass and one a head step, the same bits"; draft 1.74→1.71 / 1.76→1.68 ms
     per 5-draft window, fixed windows 8530/8502→8504/8485 ms per 512 tokens. Number in commit message only.
   - grim: norm+quant fusion with DIFFERENT bytes was rejected; this variant keeps the same quantizer/scale for all
     consumers, so it passes the byte-identity precondition. Requires a shared q8_1 scratch per drafter stage and
     kernels that accept pre-quantized rows.

8. **2-bit drafter-head screen + exact-rescore; bit-plane interval variant for exact verify** - `FN`, decode. **NEW, lower priority.**
   - Code: `include/basalt/kernels/head_proxy.cuh:19-21,38-90,103-346` (`kPer=4`, `kMaxCand=64`, `kDelta=5`,
     half-UNSHRINK `sqrt(ww/wr)`, DP4A score, radix top-k, exact rescore in `native_mmvq` order);
     `include/basalt/kernels/q80_screen.hpp:12-24`; `src/kernels/cuda/q80_screen.cu:176` supports ONLY `n_in==3072`;
     116 MB copy (`docs/models/hy4-flash.md:97`). Exact verify variant `include/basalt/models/qwen4exp/kernels/q80_vhead.hpp:1-43`
     (interval `c ± w`, `w≈1.5Σ|y|`) — 350 vs 439 µs/window (`docs/precision/qwen38-flash-next.md:188-203`).
   - grim: no head screen, no cut-vocab; `gpu_argmax_into_dev` already gives full-vocab exact argmax. Screen only
     accelerates DRAFT proposals (cannot change output); width hard-wired to 3072, CUDA intrinsics throughout.
     Prototype only if draft-head cost dominates after `may_verify`.

9. **Profile-ranked expert seeding + prompt-histogram seed** - `FN`, cold-start TTFT. **PARTIAL.**
   - Code: `data/profiles/{qwen4exp-48x512,hy4flash-42x512,...}.prof`, loader `src/core/expert_tier.cpp:630-645`;
     prompt seed `models/qwen4exp/serve/tiers.cpp:16-42` (`kSeedTokens=64`, `kSeedKeep=0.3`, top-32).
   - grim: P3 has `ResidentProfileFreq::from_trace` but unwired (Appendix D1). Idea overlaps Avalon's shipped
     `moe-cache-ranking.bin` (Appendix C1) — use Avalon's format if adopting; histogram seed is a small add-on.

10. **Priced CPU/peer expert split + peer adapt** - `FN`, prefill + two-GPU. **PARTIAL / medium.**
    - Code: `include/basalt/prefill/peer_share.hpp:90-145` (`peer_take` minimizes max of link/rows costs),
      `core/peer_experts.cpp:588-663` adapt (usage≥2, gain 1.5, capped swaps), `tier_policy.hpp:123-175` DP
      `split_by_bytes`. Algorithms neutral; `cudaMemcpyPeerAsync`/NVML transport not.
    - grim: P12 prefetch stub + P3 unwired; two-GPU pipeline exists but no priced expert split. Medium priority,
      feeds item 3 / item 6.

### E2. N/A or DUP (recorded, not promoted)

- **Engram early-row prefetch during draft** (`models/hy4flash/session.cpp:265,1575-1604`): neutral overlap pattern
  but grim's two target models have no Engram table. N/A.
- **NVFP4/W4A4 TMA tensor-core prefill** (`prefill/fp4_dense.cu`, `moe_fp4.cu`, `nvfp4_planes.hpp`): strongly
  Blackwell-specific; grim's W4A4 is gfx12 `v_dot8` only and its own quant formats differ. NO.
- **`--route-prior` / `--route-tail-skip`** (`route_flags.cpp:16-77`, `route_tail_skip.cu:14-52`, `router.cu:155-210`):
  DUP of Avalon A24 / PLAN-astrum P3; exact semantics (selection biased via `exp(Λ)`, weights unchanged; tail drop iff
  held by no GPU and every entry rank≥R, no renormalisation) corroborate Appendix C4. Keep grim's lossy gate plan.
- **Cut-vocab MTP head + 8192 hot embedding rows** (`qwen4exp/decode/mtp/head.cpp:15-167`,
  `native_head.hpp:54-77`): NEW but pack/tokenizer-dependent, needs converter support. Lower priority.
- **KV int8 + fp16 scale/64 / fp8 power-of-two scale** (`kv_q8.hpp:1-40`): DUP of grim's kvquant stack + standing
  "quantize storage, never attention math" convention.
- **Kernel micro-tiling (`__dp4a`, `__funnelshift_r`, `cp.async.bulk.prefetch.L2`)**: CUDA-only; only the neutral
  ideas (one-round tile load, aligned-word codes, compile-time width, doorbell-gated prefetch) carry over.

### E3. Effect on the ranked findings

- Item 4b (MTP micro-transfers): **promote `may_verify` to first sub-step** - pure Rust, output-neutral, no GPU gate
  needed for soundness. Row-quantize-once is second.
- Item 3 (expert ring / residency): add **use-capped victim selection** and **bounded lend with hot-slot compaction**
  as targeted additions; both are host-side algorithms that can live in the existing unwired `expert_prefetch_stream`
  / partial P3 scaffolding.
- Item 7 / graph work: add **device-side window metadata** so one captured graph can replay across positions;
  pairs with grim's graph-ON default and avoids the re-capture/per-position host writes.
- Tier 1 item 1 (QSA): add **32-head cell-tile sharing** as the post-wiring attention optimization, byte-identity gate first.
- No change to items 2, 5, 6, 8 from Basalt. No new top-level item outranks the existing Tier-1 list.

### E4. Honesty notes

- Basalt is a **different backend (CUDA/sm_120)**; numbers are its hardware's and were NOT reproduced here. All
  transfers are algorithmic, with HIP reimplementation cost.
- Several headline numbers are commit-message-only (`15 launches fewer`, `2x41 launches`, `2290→1778 us`,
  `10-15% fewer misses`); the agents flagged exactly which ones are not present verbatim in code/docs. Do not quote
  them as measured facts in grim.
- No files were written outside the review; `old/repo/basalt` is a clean depth-1 clone.

## Appendix F: llama.cpp 1.0 upstream review (2026-11-20)

Scope: `https://github.com/ggml-org/llama.cpp` cloned to `/drive/bigfast/grim/old/repo/llama1`, HEAD `e46069c`,
depth-1 (232 MB). `CMakeLists.txt:6-8` confirms `LLAMA_VERSION 1.0.0`; release `v1.0.0` published 2026-11-13. 500
commits + release notes mined; three code-verification agents read the final tree. **The clone has ONE commit, so
PR-by-PR diffs are not available; every claim below is from reading current code or release text, not a diff.**

Structural finding: upstream 1.0 now contains **`src/models/qwen4exp.cpp` (1513 lines)** with QSA, GDN, PLE, MTP,
MoE and HC, and an **upstream MoE cache** (`src/llama-moe-cache.cpp`, 984 lines). A chunk of what Avalon built as a
fork is now upstream. Consequence: for grim's Tier-1 item 1 (QSA wiring), upstream 1.0 is a **second, better-maintained
reference** than Avalon; for P3 (expert residency) upstream's ClassLRU/CRF is a stronger algorithm than grim's partial
P3 scaffolding. Paths below are relative to `old/repo/llama1/` unless marked grim.

### F1. Dense / 27B inference improvements (explicitly requested — Qwen3.8-27B is dense)

Upstream's dense model is `src/models/qwen35.cpp` (27B family). Transferable items:

1. **MTP target hidden state as a device buffer with event-ordered row copies** - `27B`, decode. **HIGH**.
   - API `include/llama.h:1172-1221`: `llama_set_embeddings_nextn_dev` (`:1210`, F32 `[n_embd, n_batch+n_seq_max]`
     on output device), `llama_get/set_embeddings_nextn_rows` (`:1213-1214`), `llama_set_ctx_state_src` (`:1218`),
     `llama_set_nextn_layer_offset` (`:1184`); batch ext row indices `:1055-1074`.
   - Impl `src/llama-context.cpp:1426-1442` (flag changes require scheduler reserve), `:1558-1619` (allocation on
     `model.dev_output()` with `ev_h_nextn_written`/`ev_h_nextn_read`, host fallback on any failure), `:1638-1661`
     (row get/set wait on the right event), `:1663-1692` (`supports_state_row` rejects cross-GPU buffers).
   - grim: `decode_graph.rs:6152` "nextn hidden copy", `:6325-6345,6789-6807` `h_src` behind `mtp_pre_norm_h_enabled()`
     (default OFF); adapter has vocab-sized D2H logits. Mirror the API shape: one state-source context + row indices +
     event ordering, with reserve invalidation when the flag changes (upstream comment `:1451-1452`: changing
     embedding flags without reserve drops acceptance to zero). **Dual-GPU caveat**: upstream rejects cross-GPU state
     rows; grim's 47/17 split must validate before relying on the device path or it silently falls back to host.
   - Upstream also supports chained NextN heads via `set_nextn_layer_offset` (`common/speculative.cpp:2052,2065,2172,2264`)
     — grim tested chaining and found no win; capability is not missing, workload result stands.

2. **`fit` charges skipped NextN layers to the output-layer GPU** - `27B`, two-GPU placement. **HIGH, small fix**.
   - `common/fit.cpp:152-157` `hp_ngl_skip = load_mtp ? 0 : n_layer_nextn`; placement `:634-663` adds `hp_nsk` to
     `n_gpu_layers` and to `tensor_split[id_last]` so the loader's `i_gpu_start` (`src/llama-model.cpp:1805-1817`)
     matches. Without this every offload boundary shifts by the NextN count.
   - grim: `plan_layer_devices` (`qwen35.rs:7489`, 47/17 auto split) has no analogous skipped-head accounting; add
     before any new two-card A/B, otherwise split measurements are against a shifted layout.

3. **Per-model rope params + correct KV-shift for quantized/hadamard K** - `27B` + `FN`, correctness and prefix reuse. **HIGH**.
   - `src/llama-graph.h:1090` model member of `llm_graph_context`; `llama-graph.cpp:1777-1806` `build_rope` sources
     `model.get_rope_params(cparams, il)` (not a graph-global), indexer variant forces NEOX `n_offs=0`.
   - `src/llama-kv-cache.cpp:2113-2179`: fast in-place KV shift only for unquantized, non-F8 K with no k_norm; slow
     path casts F32 -> **undo hadamard** -> undo k_norm -> rope -> reapply k_norm -> reapply hadamard -> copy back.
     Uses `ggml_row_size(type, …)` byte strides (`:2140-2142`) so quantized K is addressed correctly.
   - Tests `tests/test-llama-archs.cpp:1583,1670-1880` compare shifted K **and logits** against a control decode in two
     halves, with LINEAR scaling injected to catch wrong base/scale — same shape as grim's control-pair gate.
   - grim: hadamard golden exists (`hadamard.rs`, Gate D) but rope params are not per-model/per-layer in all paths;
     enforce undo/reapply around rope and byte-stride state copies for q8_1 default arena.

4. **Graph-result and scheduler reuse (decode-speed lever of 1.0)** - `both`, decode. **HIGH**.
   - `src/llama-context.cpp:780-830` `sched_new()`/`sched_reserve()` reuse scheduler + graph results when
     `sched_params.can_reuse()` (node budget less/equal, same parallel/op_offload); `:1826-1899` `process_ubatch()`
     only calls `set_inputs()` on a hit (`n_reused++`), full rebuild otherwise. Two graph-result slots
     (`llama-context.h:453-457`) because output-present vs output-absent batches have different graph-cache keys.
     `llama-graph.cpp:1622-1652` `can_reuse()` checks all input objects; env `LLAMA_GRAPH_REUSE_DISABLE`.
   - `llama-graph.cpp:1544-1584` `reset()` reuses one metadata buffer allocated at worst case and **does not zero
     unused pages** (comment `:1570`); `set_outputs()` `:1592-1620` marks every host-read sampling chain tensor as a
     graph output so ggml-alloc cannot alias it after a select.
   - grim: graph capture ON by default, M=64 prewarm, separate per-M execs; no topology-keyed reuse across reserves,
     and `grim-cli`/engine graph rebuild on config changes. Adopt two-variant cache + no-touch metadata + mark
     host-read sampler/argmax outputs live. Pairs with the `may_verify` and `gpu_argmax_into_dev` work.

5. **Cross-backend prefetch rule: only inputs computed by earlier splits** - `27B` two-card, prefill/decode. **HIGH**.
   - `ggml/src/ggml-backend.cpp:2152-2211`: skip inputs whose base tensor is not in a COMPUTE buffer (i.e. not yet
     produced by an earlier split) and skip tensors the current split writes; event per backend, next split waits
     (`:2107-2112`). Avoids copy-callback racing prefetch.
   - grim: cross-device hops in `scythe2.rs:810`, `model_loader.rs:669-671`; apply "issue boundary activation copy
     only from a stage that has produced it, gated by that stage's event" for pipeline Stage A/B work.

6. **Fixed rocBLAS workspace per handle** - `27B` + `FN` on ROCm. **MEDIUM, partly known in grim**.
   - `ggml/src/ggml-cuda/common.cuh:1392-1400,1444-1461`: one fixed 32 MiB workspace for HIP (version-gated) set once
     at handle creation; freed once (`ggml-cuda.cu:1005-1006`).
   - grim handles rocBLAS status 5 by allocator drain/retry + custom-HIP fallback (`roc_device.rs:2605-2652`) and sets
     `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` (`blaslt.rs:466-597`); prefer a preallocated cached workspace to
     remove the per-call allocation and the failure path.

7. **Few-row MMQ (2-24 rows) and empty-tile skip** - `27B` decode/prefill. **CUDA-only kernels, neutral ideas**.
   - `ggml/src/ggml-cuda/mmq-few.cu:19-20,872-905` (`MMQ_FEW_MAX_NCG=3`, `NCOLS=8`, up to 24 columns; Q4_K min 4,
     IQ4_XS min 6), dispatch `ggml-cuda.cu:2579-2588,2653-2654`; HIP/MUSA stub aborts (`:935-941`).
     `mmq.cuh:1323-1370` skips empty stream-k fixup tiles. `ggml-cpu/ops.cpp:11134-11285` fused GDN K=1 writes
     directly, K>1 via scratch — upstream removed the non-fused GDN graph expansion entirely.
   - grim already has rows_r2 dual-row and M=2 wins; the transferable rule is batch-size-independent fusion decisions
     (stable graph topology) and skipping empty tiles; do NOT port the CUDA kernels. Keep `GRIM_GDN_WY_CHUNK` as an
     oracle/reference, not a maintained fast path, now that upstream is fused-only.

8. **FP8 KV caches** - `27B` only with verification; **BLOCKED for Flash-Next**. **MEDIUM**.
   - `GGML_TYPE_F8_E4M3` in `kv_cache_types` (`common/arg.cpp:275-286`), defaults f16; per-element saturating cast
     (`ggml-quants.c:416-419`) with optional per-layer F32 scales from GGUF (`llama-model.cpp:1856-1862`) folded as
     `q*=s, k/=s` and `v/=s, out*=s` (`llama-graph.cpp:3439-3469`). On-the-fly dequant in FA
     (`fattn-common.cuh:150-172,438,688-713`). **Important correction**: `ggml-cuda/fp8.cu` is an FP8 matmul, NOT KV
     quant, and is CUDA-only. `llama-context.cpp:4487-4503` explicitly **rejects FP8 KV for LLM_ARCH_QWEN4EXP**
     (and DFLASH, MLA/DeepSeek families); qwen35 is not in that rejection list, but verify a real checkpoint before
     assuming support. grim's q8_1/int8/WhiteCrow stack is different; the portable idea is Q/V scale folding.

### F2. Flash-Next / MoE improvements

1. **Upstream MoE cache with ClassLRU + CRF and CPU share** - `FN`. **PARTIAL in grim (P3 unwired), algorithm NEW**.
   - `src/llama-moe-cache.cpp` (984 lines): `moe_cache_lru` `:21-288` uses n_class=8, `crf_half_life=128` tokens,
     CRF = decayed uses +1, class=floor(log2(CRF)), victim=max(age/2^class) over per-class LRU heads (`:117-176`);
     `plan()` `:187-287` keeps hits, ranks misses by CRF, sends lowest-CRF misses to CPU when cache full, one zero
     slot per token max. `impl` `:321-879`: max_batch 32, CPU share in 1/65536 with carry (`:388,813-818`), GPU
     required, tensor-parallel rejected (`:412-420`), only fully host-resident layers cached (`:422-445`), bank has
     `n_zero+1` slot for MMQ overread (`:555-559`).
   - No CPU wait on cached experts: build_lookup `g.cb(slots,...)` (`:891-947`), `copy()` uploads async and rewrites
     slot map (`:767-862`), `build_experts` adds GPU-cached + CPU-miss paths (`:949-963`).
   - API/flags: `include/llama.h:409-410` `moe_cache_size`/`moe_cache_cpu_share`, `--moe-cache-mib` /
     `--moe-cache-cpu-share` (`common/arg.cpp:2751-2771`), defaults 0/0.0 (`common/common.h:619-620`); doc
     `docs/moe-cache.md`, tuner `scripts/moe-cache-tune.py`.
   - grim: P3 primitives exist but `from_env()` has no caller and Flash-Next routes on device without a prior
     (Appendix D1). Adopt ClassLRU/CRF + CPU-share fraction + zero-slot scheme; backend-agnostic, but needs the
     scheduler/copy-callback equivalent. Pipeline parallel is disabled when the cache is on
     (`llama-context.cpp:584-590`) — account for that in grim's two-GPU pipeline work.

2. **Upstream `qwen4exp` as a QSA/GDN/MTP/PLE reference** - `FN`. **REFERENCE, not a new lever**.
   - QSA: hparams `:64-83` (single `compress_ratio`, `kpool_row=2`, `by_order=true`, `select_tail=true`);
     `build_inp_kpool :718-811`, `build_qsa_sel :813-950` (pooled-key lightning indexer + `ggml_top_k` +
     `ggml_top_k_mask` at `:891`), `build_attn_qsa :951-1010`, dispatcher `:1011-1087`. Device-side fused mask via
     `llm_graph_context::build_fused_top_k_mask` (`llama-graph.cpp:3653-3670`) with `fused_top_k_mask` default ON
     (`llama-context.cpp:346`, probe `:744`); kernel `ggml/src/ggml-cuda/top-k-mask.cu`.
   - GDN `:1088-1218` (conv, l2 norm, recurrent attn; sigmoid gate not silu vs Qwen3.5); PLE hash `:1298-1365`;
     MTP `graph_mtp :582-671`; MoE + shared expert `:300-307`; HC mix/combine `:352-450`.
   - This is the concrete template for report item 1's stage 2: replace grim's host `to_vec_f32`/host mask
     (`qwen4exp_flash_next.rs:1504-1710`) with device top-k + fused mask + selected-cell attention. No explicit
     tail-skip/route-prior symbol in the file — do not claim one.

3. **External PLE/engram shards + lazy mmap + row prefetch** - `FN`, cold load/TTFT. **MEDIUM**.
   - `--ple` == `--shard` (`common/arg.cpp:3039-3048`); loader `src/llama-model-loader.cpp:600-758` recognizes a
     one-tensor `per_layer_token_embd.weight` shard (`:647-649`), skips split numbering checks, and lets it override
     the backbone copy (`:705-706,726-734`). Lazy tensor + optional F32 scale (`qwen4exp.cpp:216-219`), all files
     mapped when mmap OR any lazy tensor (`:1505-1536`), `llama_prefetch_rows` warms selected hash rows
     (`qwen4exp.cpp:1362-1364`).
   - grim's Flash-Next plan already puts the PLE table in host RAM; adopt override/mmap/prefetch mechanics if the
     checkpoint arrives as separate shards. Hash constants come from GGUF KV (`src/llama-arch.h:377-385`).

4. **Reference-counted checkpoints + parent-before-child scheduling** - `both`, multi-request TTFT. **HIGH**.
   - `tools/server/server-slot-sched.h:13-46`: checkpoints are `shared_ptr<const common_prompt_checkpoint>`; clone
     copies the pointer list, not MiB blobs. `server-slot-sched.cpp:13-58` `get_reuse()` newest-first with SWA-aware
     validity (`n_swa`). `tools/server/server-context.cpp:3378-3394` `iterate()` does two passes: non-child slots
     first, child slots second; children wait (`SLOT_STATE_WAIT_OTHER`) until parent publishes the shared prefix
     (`:4304-4312,4727-4743`).
   - grim: radix prefix cache default-on (`engine/core.rs:289-294`), P10 checkpoint cache partial/unwired (Appendix D1).
     Adopt `Arc`-shared immutable blobs, newest-first + SWA validity, and parent-first two-pass scheduling.

5. **Graph node budget from model structure + MoE reserve** - `FN`/`27B`, stability. **MEDIUM**.
   - `src/llama-model.h:1054-1056` `graph_nodes_per_tensor()` (8 default, 32 for most large/MoE models, 64 for
     selected); `llama-context.cpp:2913-2935` `graph_max_nodes()` adds `+32*n_layer_all` when `moe_cache_size>0`
     and sampler-chain nodes; failure to reserve causes realloc churn (tracked `n_reallocs_same/diff` at `:3092-3109`).
     Node budget moved out of `n_ubatch` into model classes (`#32648`), and a per-layer MoE reserve added (`#32636`).
   - grim's M=64 prewarm depends on a stable graph topology; budget from static model structure + MoE terms, not from
     the current batch.

### F3. DUP / N-A / not promoted

- MoE CPU-miss + LRFU idea: DUP of Avalon A30/A31, Basalt E1.2, and PLAN-astrum P3 — upstream's ClassLRU/CRF is a
  better-specified version of the same tactic, not a new lever.
- QSA tile lists / GPU mask: DUP of Avalon A13-A15; upstream adds a fused `TOP_K_MASK` op absent from Avalon.
- `fp8.cu` FP8 matmul: CUDA-only, excluded on HIP (`:3`); note only.
- Few-row MMQ, stream-k fixup, NVFP4 epilogue scale fusion, Vulkan FA / RDNA prompt FA (#31323): backend-specific
  kernels; ideas (batch-size-independent fusion, empty-tile skip) noted, no direct port.
- `mtmd` shared compute buffer/worst-case warmup (`tools/mtmd/README-dev.md:36-37`): neutral memory idea, low priority.
- New upstream architectures (Kimi-K3.1-Pro, Ripple, xing4_0, drama-3): outside grim's two target models.

### F4. Effect on the ranked findings and honesty notes

- Item 1 (QSA): upstream 1.0 `qwen4exp` becomes the **primary** wiring reference; Avalon tile lists remain the
  performance follow-up. Add fused `TOP_K_MASK` to stage 2 scope.
- Item 3 (expert ring/residency): add upstream ClassLRU/CRF + CPU-share + zero slot as the concrete replacement for
  grim's unwired P3; remember pipeline parallel was disabled upstream with MoE cache on.
- Item 4b / MTP: `may_verify` (Basalt) + device state rows (upstream) are the two complementary pieces; device path
  must be validated for grim's cross-GPU split because upstream rejects cross-GPU state rows.
- Item 5 (pinned checkpoint): upstream's `Arc`-shared checkpoints + parent-first scheduling strengthen the TTFT case;
  still need Avalon's pinned capture for the first-token copy.
- Dense/27B: items 2 (device hidden), 4/4b (graph reuse, argmax), 6 (two-GPU placement), and new F1.3 rope/KV-shift
  are the dense-specific additions. Do not let MoE/QSA work crowd out the 27B path — it is a dense model.
- Honesty: llama1 is a **single-commit snapshot**; PR numbers label features but no diff was available. Release-note
  and code claims were not benchmarked here. FP8 KV is blocked for QWEN4EXP in upstream, and `fp8.cu` is not a KV
  quantizer. No files written outside the review; `old/repo/llama1` is a clean depth-1 clone.

## Appendix G: dense-model (27B) improvements in Avalon vs Basalt (2026-11-20)

User question: do Avalon or Basalt contain inference improvements for DENSE models? Two dedicated code verifiers
checked both trees. Paths relative to their clones unless marked grim.

### G1. Avalon — verdict: headline work is MoE-only; a few generic changes do reach dense

- Dense target in Avalon = `LLM_ARCH_QWEN35` -> `llama_model_qwen35` (`src/llama-arch.h:47`,
  `src/llama-model.cpp:333-334`). Dense FFN is asserted: `src/models/qwen35.cpp:481-490` checks
  `ffn_gate_inp == nullptr` and builds SILU FFN; GDN + full attention mix at `:158-170`.
- **MoE-ONLY, cannot help 27B**: expert cache (requires `_exps`/`_chexps` 3-D host tensors,
  `src/llama-context.cpp:882-907`, mode forced OFF otherwise `:984-987`), hosted MoE, expert prefetch ring
  (`moe-ring.h`), RAM arena (`moe-arena.h`), QSA tiles (`qwen4exp.cpp:1357-1397`, zero QSA calls in `qwen35.cpp`),
  fused `hc_post`+norm (qwen35 uses `build_norm_gated` at `qwen35.cpp:257-265`, not `ggml_hc_post`).
- **Generic / shape-gated, DOES reach dense**: backend greedy argmax with `data->logits=nullptr`
  (`src/llama-sampler.cpp:1100-1133`, NVIDIA default per `README.md:236`); fused GDN kernels
  (`cparams.fused_gdn_ar/ch` default true `src/llama-context.cpp:368-369`, consumed by dense qwen35 at
  `qwen35.cpp:452` via `delta-net-base.cpp:433-439`); GDN WMMA chunked prefill (shape gate S=128/HK=16/HV=48,
  single seq, >=128 tokens — `ggml-cuda/gdn-wmma.cu:1264+`, NOT architecture-gated, but 27B dims unverified);
  pinned checkpoint spare buffer (>16 MiB heuristic, `common/common.cpp:2510-2538`); IQ3_KS quant (only if weights
  stored that way); `top_k_radix_row` likely generic (gating not fully confirmed).
- **Conclusion**: README's "other models run as in llama.cpp" is overstated in the literal sense but correct for
  all headline optimizations. No MoE/cache/QSA/hc idea leaks into dense. For grim 27B the only clearly active
  Avalon dense items are backend argmax (item 4) and fused GDN (item 2 / Appendix E7); GDN WMMA is the same
  shape-gated candidate already listed in item 2. No NEW dense item from Avalon.

### G2. Basalt — verdict: substantial, model-specific dense/27B work; several transferable ideas

All numbers below live in commit messages or `docs/models/qwen38-27b.md` / `docs/precision/qwen38-27b.md`; none were
re-benchmarked here. Paths under `src/dense/`, `src/models/qwen35/`, `include/basalt/dense/` unless noted.

1. **`--kv-resident` two-tier KV** - **NEW to grim, idea neutral, mechanism CUDA-specific**. Flag parse
   `src/models/qwen35/adapter.cpp:652`, default 0 (`include/basalt/program/options.hpp:58-60`), rule
   `src/dense/model.cpp:322-323`; pinned mapped host tier `cudaHostAlloc(...Mapped|Portable)` at
   `session.cpp:63-93`, `far_view` partition `include/basalt/kernels/gqa_attn.hpp:34-44`; decode reads far cells in
   `gqa_decode_attn` via `forward.cpp:663`; prompt staging `forward.cpp:362-399`. Commit `5f23040a` reports a
   250k-token prompt (4345->4171 tok/s) and hit to host bandwidth (~24 of 28.8 GB/s) — decode with the far tier was
   MUCH slower (45.4->3.1 tok/s plain), so this is a capacity/TTFT lever, NOT a free decode win. grim has no pinned
   KV-host tier; q8_1 changes byte math, not the design. Adopt only with explicit near/far A/B.

2. **Pipelined layer split for long prompts** - **PARTIAL/dup**, concept maps to grim's 47/17 split. Planner
   `src/dense/model.cpp:358+`, `qwen35/adapter.cpp:95+,629-644`; `Stage::read` `src/dense/stage.cpp:132`,
   `split_read :985-1094` (main reads chunk i+1 while stage reads chunk i, residual via pinned RAM, cells returned
   per chunk); decode hand-off `dense/internal.hpp:113-121`, `forward.cpp:609-645,725-733`, graph per card `:688`.
   Commit `d9408db3` reports decode regression on the split (~0.37 ms/layer on a 5060 Ti) and "4K prompt = one
   chunk, no overlap" — useful caveat for grim's own pipeline A/B.

3. **16-row decode windows + automatic draft width (`--spec` 0..16, default 16)** - **NEW, neutral heuristic,
   row count not portable as-is**. `dense/internal.hpp:23-25` (`kWindow=16`, `kBatch=16`, `kNarrow=8`),
   `session.cpp:200-204,317-401`, auto width `wide_ = n >= max(2, block-2)` at `:744`, `:208`, flag default
   `qwen35/adapter.cpp:690`. Commit `20e8c370` reports struct 506->605 tok/s, 10.1->12.8 tok/window. grim's GEMV
   regime is narrow (M=1/2 with rows_r2, M=64 prewarm); 16 rows moves past the M=1/2 crossover — re-measure before
   adopting any width heuristic.

4. **DFlash2 block drafter** - **NEW algorithm, needs a drafter pack**. Pack shape `include/basalt/dense/model.hpp:63-81`,
   `qwen35/adapter.cpp:28-53` (block 2..8, top_k<=20); pass `forward.cpp:1066-1177` (row_logprobs over all block
   candidates, `CandidateDraw`, `dyn_conv2`, selector); offer-all + confidence window `session.cpp:222-228`; sampled
   drafts `session.cpp:821-834` with exact `min(1,p/q)` (`kernels/cuda/sampler.cu:345`); per-slot share `session.cpp:272`.
   grim has only native MTP (default OFF) and already implements `min(1,p_target/p_draft)` (`speculative_wrapper.rs:462`).
   Offer-all + confidence + share are neutral; weights/kernels are not.

5. **`window_gemm` fused epilogues + PDL + batched windows + GDN overlap + per-state graphs** - **PARTIAL/NEW ideas**.
   `include/basalt/kernels/window_gemm.hpp:1-90` (<=16 rows, `Epilogue{Store,Add,AddBf16,SwiGLU}`, <=3 matrices/call);
   call sites `forward.cpp:41,62-80,134-185`, `PdlScope` `:655,784,799,819,1104`; batch `:773-901` (graph-per-composition
   LRU 64, `internal.hpp:148`); GDN commit deferral `session.cpp:714-766` with `Join` guard `:216-219`; per-state
   graphs `internal.hpp:188-193,207`, captured `forward.cpp:686-710`. PDL is CUDA-only (`kernels/pdl.hpp:56-77`,
   gated CC>=9.0/CUDA>=12.3) — do NOT port to HIP. Neutral: fused epilogues, LRU graph-per-composition, deferred
   GDN commit, per-state graph copies. grim's M=2 GEMV makes a 16-row fused GEMM less attractive today.

6. **Prompt chunks: 4096 (dup) + BF16 residual + cells-only last layer (NEW, neutral)** - `internal.hpp:28-32`
   (`kChunk=4096`, `kAttnSplits=3`), loop `session.cpp:416-428` — grim already defaults `GRIM_PREFILL_CHUNK_SIZE=4096`.
   BF16 residual with single rounding `forward.cpp:335,916-936`, fp32 add in epilogues `:149-160`
   (commits `2509f5da`, `d36c2a0a`). Cells-only last layer `forward.cpp:942-972` / `:418-429` (skips one of 16 prompt
   attentions, 94 M FP8 weights, 1 of 64 MLPs per chunk — `12516e77`). Both are architecture-neutral and fit grim's
   prefill path; NVFP4 W4A4 planes (`forward.cpp:409-417,486-499,531-540`, `prefill/fp4_dense.cu`) are
   Blackwell/TMA-only — N/A.

7. **Batched requests + drafter context in running state** - **NEW neutral principle, currently dormant in grim**.
   `session.cpp:340-401` (`decode_batch`), `:255-289` (`draft_batch`, per-slot share `:272`), graph `forward.cpp:848-901`;
   checkpoint includes block drafter ring + valid-length (`session.cpp:486-539`, ~42 MB per commit `31e6819e`).
   Principle: serialize drafter ring and valid cell count in the running state so a prefix-cache hit restores it;
   grim's MTP is OFF so nothing to restore today, but the same pattern applies when/if MTP is enabled.

### G3. Effect on the ranked findings

- **Avalon adds no new dense item**; its dense-relevant pieces were already in item 2 (GDN WMMA) and item 4
  (backend argmax). The "Avalon = Flash-Next-only" framing is safe for all transfer recommendations.
- **Basalt adds three dense ideas worth acting on**: (a) BF16 residual + cells-only last layer in the prefill path
  (neutral, cheap, pairs with M=64 prewarm); (b) earlier-split-only cross-card prefetch + per-chunk pipelining details
  for the 47/17 split (already in Appendix F1.5/F2.2); (c) `--kv-resident` two-tier KV as a *conditional* capacity
  lever with a documented decode penalty — do not present it as a speed win.
- **Do not promote**: 16-row windows / window_gemm / PDL / DFlash2 block drafter as direct copies — they assume a
  different GEMV crossover and a drafter pack grim does not have. Keep the neutral heuristics (offer-all, confidence,
  per-slot share, deferred commit) as design references only.

### G4. Honesty notes

- No GPU execution in either verifier; all speed numbers are commit-message/doc claims. Basalt's own `--kv-resident`
  commit documents a large decode slowdown when reading the far tier — report that, not only the prefill gain.
- Avalon's `top_k_radix_row` genericity and 27B GDN-WMMA dimension match were not fully confirmed; flagged.
- No files were written outside the review; both clones remain clean depth-1 checkouts.
