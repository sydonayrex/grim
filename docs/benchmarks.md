# Decode GEMM Benchmark Report: `grim_decode_gemm_f16` vs. `rocBLAS` vs. `WMMA`

## 1. Executive Summary

During inference serving, the autoregressive decode phase evaluates batch sizes $M \in \{1, 2, 4, 8\}$ against large model projection weights ($N, K \in [4096, 28672]$). Because $M \ll K, N$, decode GEMMs operate in an intensely memory-bandwidth-bound regime rather than a compute-bound regime.

This document compiles the benchmark results evaluating the three candidate execution paths on AMD RDNA4 hardware (`gfx1201`, AMD Radeon RX 9070 XT):
1. **`grim_decode_gemm_f16`**: Custom hand-written HIP kernel doing direct row/column dot products with FP32 accumulation and 256-thread blocks.
2. **`rocBLAS`**: Vendor-optimized `rocblas_gemm_ex` with FP16 inputs and FP32 compute (`ROCBLAS_GEMM_FLAGS_NONE`).
3. **`WMMA`**: Wave Matrix Multiply-Accumulate tensor-core kernel (`grim_wmma_gemm`) using $16 \times 16$ tile fragments.

### Key Takeaways
- **rocBLAS decisively outperforms `grim_decode_gemm_f16` across all tested decode shapes**, yielding **1.23x to 5.33x speedups** (reducing kernel execution latency by up to **81%** on down-projections).
- **WMMA is structurally unsuited for unpadded $M < 16$**: Because WMMA hardware matrix cores operate on minimum $16 \times 16 \times 16$ fragments, unpadded execution with $M < 16$ computes cross-row dot products and invalid outputs unless explicitly padded to 16 rows.
- **Dispatch Decision**: The decode GEMM dispatch arm in `device_compute.rs` ($M \le 8, F16$) and the HIP graph capture manager in `roc_device.rs` were re-routed to `launch_rocblas_gemm_f16`.

---

## 2. Test Environment & Methodology

- **GPU**: AMD Radeon RX 9070 XT (`gfx1201`)
- **ROCm Runtime**: ROCm 7.x
- **Compute Peak**: ~6.8 TFLOPS FP16 (calibrated)
- **Theoretical Peak Bandwidth**: ~515 GB/s
- **Precision**: Inputs $A$ and $B$ in FP16 (`half::f16`), accumulation in FP32, output in FP16.
- **Timing Methodology**: 15 warmup iterations, 100 timed iterations measured with device stream synchronization (`Instant::now() / iters`).
- **Memory Bandwidth Metric**:
  $$\text{Effective BW (GB/s)} = \frac{(M \cdot K + K \cdot N + M \cdot N) \times 2 \text{ bytes}}{\text{Latency (s)} \times 10^9}$$

---

## 3. Benchmark Results

### 3.1 Comprehensive 3-Way Shape Comparison Table

#### Single-Token Decode (M = 1)

| Target | Projection | Shape (M x K x N) | Decode (µs) | rocBLAS (µs) | WMMA-T R3 (µs) | WMMA-T R4 (µs) | rocBLAS BW | WMMA-T R3 BW | WMMA-T R4 BW | R3 Parity | R4 Parity |
|:-------|:-----------|:-----------------:|------------:|-------------:|---------------:|---------------:|-----------:|-------------:|-------------:|----------:|----------:|
| 8B     | Q/K/V/O    | 1 x 4096 x 4096   |     1372.3  |       437.2  |      **292.2** |          343.9 |  76.8 GB/s |**114.9 GB/s**|    97.6 GB/s |   6.10e-5 |   6.10e-5 |
| 8B     | Gate/Up    | 1 x 4096 x 14336  |     1590.2  |      1035.1  |      **443.1** |          588.3 | 113.5 GB/s |**265.1 GB/s**|   199.7 GB/s |   1.53e-5 |   1.53e-5 |
| 8B     | Down       | 1 x 14336 x 4096  |     4955.3  |       943.4  |      **626.5** |          652.1 | 124.5 GB/s |**187.5 GB/s**|   180.2 GB/s |   2.44e-4 |   2.44e-4 |
| 70B    | Q/K/V/O    | 1 x 8192 x 8192   |     3103.8  |       974.7  |      **586.8** |          962.8 | 137.7 GB/s |**228.8 GB/s**|   139.4 GB/s |   6.10e-5 |   6.10e-5 |
| 70B    | Gate/Up    | 1 x 8192 x 28672  |     2957.7  |    **2242.3**|         3274.3 |         3769.4 |**209.5 GB/s**|143.5 GB/s  |   124.6 GB/s |   3.05e-5 |   3.05e-5 |
| 70B    | Down       | 1 x 28672 x 8192  |     7790.8  |      2280.0  |     **1390.0** |         1719.9 | 206.1 GB/s |**338.0 GB/s**|   273.2 GB/s |   1.22e-4 |   1.22e-4 |

#### Batched Decode (M ∈ {2, 4, 8}, Llama-3 8B Shapes)

| M | Projection | Shape (M x K x N) | Decode (µs) | rocBLAS (µs) | WMMA-T R3 (µs) | WMMA-T R4 (µs) | rocBLAS BW | WMMA-T R3 BW | WMMA-T R4 BW | R3 Parity | R4 Parity |
|:-:|:-----------|:-----------------:|------------:|-------------:|---------------:|---------------:|-----------:|-------------:|-------------:|----------:|----------:|
| 2 | Q/K/V/O    | 2 x 4096 x 4096   |     1166.9  |       322.8  |      **213.7** |          241.3 | 104.0 GB/s |**157.2 GB/s**|   139.2 GB/s |   6.10e-5 |   6.10e-5 |
| 2 | Gate/Up    | 2 x 4096 x 14336  |     1591.1  |      1057.0  |      **439.2** |          574.9 | 111.2 GB/s |**267.6 GB/s**|   204.4 GB/s |   1.53e-5 |   1.53e-5 |
| 2 | Down       | 2 x 14336 x 4096  |     4930.5  |       939.2  |      **622.4** |          663.2 | 125.1 GB/s |**188.8 GB/s**|   177.2 GB/s |   2.44e-4 |   2.44e-4 |
| 4 | Q/K/V/O    | 4 x 4096 x 4096   |     1317.8  |       351.4  |      **253.0** |          260.1 |  95.7 GB/s |**132.9 GB/s**|   129.3 GB/s |   2.44e-4 |   2.44e-4 |
| 4 | Gate/Up    | 4 x 4096 x 14336  |     1519.6  |      1045.3  |      **464.0** |          599.6 | 112.5 GB/s |**253.4 GB/s**|   196.1 GB/s |   2.44e-4 |   2.44e-4 |
| 4 | Down       | 4 x 14336 x 4096  |     4913.6  |       971.6  |          665.1 |      **625.4** | 121.0 GB/s |  176.8 GB/s  |**188.0 GB/s**|   4.88e-4 |   4.88e-4 |
| 8 | Q/K/V/O    | 8 x 4096 x 4096   |     1281.7  |       386.5  |          339.6 |      **339.1** |  87.2 GB/s |   99.2 GB/s  | **99.3 GB/s**|   2.44e-4 |   2.44e-4 |
| 8 | Gate/Up    | 8 x 4096 x 14336  |     2999.7  |       926.0  |      **486.4** |          576.8 | 127.1 GB/s |**242.0 GB/s**|   204.1 GB/s |   2.44e-4 |   2.44e-4 |
| 8 | Down       | 8 x 14336 x 4096  |     4749.9  |       876.2  |          757.0 |      **720.9** | 134.4 GB/s |  155.5 GB/s  |**163.3 GB/s**|   4.88e-4 |   4.88e-4 |

#### Summary by Kernel Candidate

| Kernel Candidate                      | Architecture Support | Tile Strategy       | Mean Latency (8B M=1) | Bandwidth Range | Verdict                 |
|:--------------------------------------|:---------------------|:--------------------|----------------------:|:----------------|:------------------------|
| `grim_wmma_gemm_b_transposed` (R3)    | RDNA3 & RDNA4        | Single-Wave 16x32   |            **453 µs** | 114 - 338 GB/s  | **Overall Winner (SOTA)**|
| `grim_wmma_gemm_b_transposed_rdna4`   | RDNA4 Native Only    | Multi-Wave 16x64    |                528 µs |  97 - 273 GB/s  | Strong for deep K       |
| `rocBLAS`                             | All GCN / CDNA / RDNA| rocBLAS internal    |                805 µs |  76 - 211 GB/s  | General fallback        |
| `grim_decode_gemm_f16`                | All ROCm             | Scalar thread dot   |               2639 µs |  24 - 160 GB/s  | Deprecated              |

---

## 4. Analysis & Observations

### 4.1 Comparison: RDNA3 Single-Wave vs RDNA4 Multi-Wave

1. **Single-Wave 16x32 (`WMMA-T R3`) Wins on Decode:**
   - On small batches ($M=1, 2, 4$), the single-wave design wins because it has **zero workgroup synchronization barriers** (`__syncthreads()`) and keeps all activation tile data in registers.
   - At $M=1$ Gate/Up, R3 hits **443.1 µs (265.1 GB/s)** vs 588.3 µs for R4 and 1035.1 µs for rocBLAS.
   - At $M=1$ 70B Down, R3 achieves **1390.0 µs (338.0 GB/s)** vs 2280 µs for rocBLAS.
2. **Multi-Wave 16x64 (`WMMA-T R4`) on Deep K ($M=4, 8$ Down):**
   - On large $K$ ($K=14336$), R4 pulls ahead slightly at $M=8$ (**720.9 µs vs 757.0 µs**) because sharing $A$ across 2 waves in LDS amortizes memory bus load for the deep inner loop.
3. **Safety Across Hardware Generations:**
   - By gating the multi-wave LDS kernel to RDNA4 (`gfx1200+`), we avoid the known cross-wave LDS barrier lockups present on RDNA3 hardware while retaining the single-wave register path as a universally compatible, ultra-fast baseline.

---

## 5. Implementation Status

1. **Direct Matmul Dispatch**:
   - Updated `RocmDevice::matmul_op` in `crates/grim-backend-rocm/src/device/device_compute.rs`.
   - The decode arm ($M \le 8, F16$) routes directly to `launch_rocblas_gemm_f16` with automatic fallback to standard rocBLAS if tuned solution indices are rejected.
2. **Graph Capture & Replay**:
   - Updated `RocmDevice::decode_graph_capture_and_replay` in `crates/grim-backend-rocm/src/device/roc_device.rs`.
   - Captured HIP graphs replay `launch_rocblas_gemm_f16` within the captured stream.
3. **Reproducibility**:
   - Benchmark executable is maintained at [`crates/grim-backend-rocm/examples/bakeoff_decode_gemm.rs`](crates/grim-backend-rocm/examples/bakeoff_decode_gemm.rs).
   - Run command:
     ```bash
     GRIM_GPU_TEST=1 cargo run --release --features gpu-test-shims --example bakeoff_decode_gemm -p grim-backend-rocm
     ```

---

## 6. BLASLt Prefill Promotion Gate — Q8 Candidate Rejection

The native-F32 BLASLt prefill integration is opt-in and remains separate from
this decode benchmark. To measure a real model prefill candidate on GPU 0, a
temporary Q8_0 candidate was tested with:

```bash
GRIM_BLASLT_PREFILL=1 GRIM_BLASLT_QUANT_PREFILL=1 \\
  grim-cli run models/LFM2.5-350M-Q8_0.gguf \\
  "What are the top 5 Greek thought experiments?" --raw --device rocm \\
  --min-tokens 195 --max-tokens 195 --temperature 0.7 --top-p 0.9 \\
  --top-k 40 --seed 42
```

The candidate dequantized Q8_0 weights to F32, retained the expanded weights
on device, and used the canonical BLASLt prefill route. Raw no-profiler
measurements on `gfx1201` were:

| Path | Samples (tok/s) |
|---|---:|
| Default Q8_0 | 297, 301, 300 |
| Q8_0 → F32 → BLASLt | 301, 300, 302 |

The candidate did not reach the `350 tok/s` promotion criterion. More
importantly, both stochastic 195-token output and deterministic greedy output
diverged from the default model. The temporary production route was therefore
removed under the plan's parity-rejection rule; it must not be enabled as a
fallback.

The `rocprofv3` launch census explains the cost. Both runs used 193
`hipGraphLaunch` calls, but the candidate increased HIP kernel launches from
1,180 to 1,366 by adding 93 each of `grim_dequant_q8_0`,
`grim_transpose_2d_f32`, and `grim_col_major_to_row_major_f32`. These counts
are launch guidance only and were not used to report throughput. A future
promotion candidate must preserve quantized arithmetic semantics or use a
native-F32 model; expanding Q8 weights to F32 is not sufficient.

---

## 7. GPU 1 Experiment Matrix

GPU 1 was an idle AMD Radeon RX 9060 XT. The release CLI was rebuilt from the
current tree, then every case was run with `ROCR_VISIBLE_DEVICES=1`, the same
raw LFM2.5-350M-Q8_0 prompt, seed 42, temperature 0.7, top-p 0.9, top-k 40,
and a forced 195-token decode window. Throughput was measured without a
profiler.

| Experiment | tok/s |
|---|---:|
| Paired default | 326–334 |
| Fused QKV | 333 |
| F16 KV | 333 |
| Eight-wave dot4 | 127 |
| Dot4 prequant | 328 |
| Residual/GateUp fusion | **341–343** |
| Sampler block 512 | 319 |
| Sampler block 256 | 309 |
| Generic GEMM control | 94 |
| Dot4 tile16 | 284 |
| Dot4 tile8 | 314 |
| Down/Q8.1 add-prequant | 322 |
| Native-F32 BLASLt flag on Q8 model | 331 |

The residual/GateUp candidate was repeated three times: default `327, 332,
329` tok/s versus candidate `343, 342, 341` tok/s. Its response matched the
paired default. Combinations did not close the gap: residual + F16 KV `343`,
residual + fused QKV `340`, residual + both `342`, residual + dot4 prequant
`343`, and residual + add-prequant `334` tok/s.

`rocprofv3` launch guidance showed 193 graph launches for both default and
residual fusion, while HIP kernel launches fell from 1,180 to 1,132. The
fused kernel `grim_dot4_add_rms_norm_gate_up_silu_q80_gemv` ran 3,120 times.
Residual/GateUp is therefore the best current candidate, but it remains about
2% below the 350 tok/s target and is not promoted.

---

## 8. Fused-Kernel Coalescing — GPU 1 Target Crossing

The fused residual/GateUp kernel originally serialized the residual row
through lane 0. The retained optimization distributes those stores across all
32 lanes:

```text
for (col = lane; col < K; col += 32)
```

The RMS reduction, Q8_0 activation quantization, dot4 accumulation, and SiLU
arithmetic are unchanged. The eight-output tile and fast-SiLU experiments were
measured but removed because they were slower or unsafe to generalize.

The final raw GPU-1 release results, with the same 195-token protocol and no
profiler, were:

| Path | tok/s |
|---|---:|
| Paired default | 328, 330, 330 |
| Coalesced residual/GateUp | **353, 354, 353** |

The final paired confirmation measured 329 tok/s default versus 354 tok/s
fused, with identical generated output. The fused graph suite passed `11/11`
with `GRIM_FUSED_RESIDUAL_GATEUP=1`, and the serialized ROCm backend unit
suite passed `468/468`. The candidate now crosses the 350 tok/s criterion on
GPU 1 while remaining opt-in pending a clean-process parity/promotion review.

---

## 9. Clean-Process Promotion Review

The promotion review cleared all generated HSACO/JIT caches and ran
`cargo clean` before rebuilding the release CLI. Default and fused runs then
used separate empty cache directories in separate processes.

The first cold-cache fused process measured 347 tok/s. Warm fresh-process
fused runs measured 356, 355, and 352 tok/s; warm default runs measured 328,
329, and 328 tok/s. Deterministic greedy output and the stochastic 195-token
response matched after excluding the hardware calibration diagnostic line.

The fused graph suite passed `11/11`. The promotion gate now enables the
coalesced residual/GateUp path by default only for `gfx1200`; other targets
retain the split path. `GRIM_FUSED_RESIDUAL_GATEUP=0` is the kill switch.
Final promoted-default warm runs measured 357, 354, and 357 tok/s, while the
kill-switch run measured 324 tok/s in the same review session.

---

## 10. Model Baseline Preparation

A model-agnostic baseline command is now available for checkpoint preparation:

```bash
# Inventory all discovered checkpoints; missing manifest entries are pending.
./target/release/grim-cli baseline --model-dir models

# Execute fixed prefill baselines for selected available checkpoints.
./target/release/grim-cli baseline \
  --model-dir models \
  --manifest /path/to/checkpoints.manifest \
  --run --device rocm --prompt-tokens 8 --warmup 1 --steps 3
```

The command discovers `.gguf`, `.grim`, `.safetensors`, and `.bin` files
recursively, loads through the normal model loader, and reports JSON containing
path, architecture, device, status, forward samples, mean forward time, and
errors. A manifest can name checkpoints before they are downloaded; unavailable
entries are reported as `pending` without failing the inventory.

Validation performed on GPU 1:

- Baseline unit tests: `3 passed`.
- Inventory mode discovered all current model files.
- Execution mode passed on `LFM2.5-230M-Q4_K_M` at `463.7 ms` mean 8-token
  forward and `LFM2.5-350M-Q4_K_M` at `608.6 ms`.
- Manifest mode reported a future checkpoint as `pending`.

These are prefill forward baselines for model preparation, not decode tok/s and
not comparable to the 350 tok/s decode promotion gate.

---

## 11. Qwen3.8 Q4_K Dual-GPU Target Audit

**No Qwen tok/s is recorded.** The 27B load now clears every previous blocker,
but generation has not completed under measurement, so nothing here is a
promotion. Commits: `10e57867` (six root-cause fixes), `69099316` (KV-quant
parity gates).

### Architecture, from GGUF metadata

`general.architecture=qwen35`, `general.name=Qwen3.8-27B`, 65 blocks,
`embedding_length=5120`, 24 query / 4 KV heads, SSM `state_size=128`,
`inner_size=6144`, `conv_kernel=4`, `time_step_rank=48`, `group_count=16`. A
65-layer Qwen3.5/3.8 hybrid with full attention every 4th layer — not the dense
Qwen3.8 Flash-Next/MoE path. Tokenizer is gpt2 BPE, `pre=qwen35`, 248,320 tokens
and 247,587 merges, `bos=248044` / `eos=248046` / `padding=248055`. The chat
template has vision/image and video branches.

The file carries **no `tokenizer.ggml.vocab_size` and no `qwen35.vocab_size`**.
The 32,000 that appeared in earlier logs was the `unwrap_or(32000)` default, not
stale metadata: the vocabulary only exists as the `tokenizer.ggml.tokens` array
length, which a scalar `get_u32` lookup can never resolve.

### KV arena sizing (why context, not weights, is the constraint)

The decode graph pre-allocates one arena per full-attention layer at full
context, before any weight is placed. With 4 KV heads × 256 head_dim, f32, K+V,
across the 17 full-attention layers:

| context | KV arenas | + 15.2 GB Q4_K weights | fits 34 GB? |
|---|---|---|---|
| 32,768 | 4.6 GB | 19.8 GB | yes |
| 65,536 | 9.1 GB | 24.3 GB | yes |
| 98,304 | 13.7 GB | 28.9 GB | yes |
| 131,072 | 18.3 GB | 33.5 GB | borderline |
| 228,000 | 31.8 GB | 47.0 GB | **no** |

Measured peak VRAM during a 228k load reached 17.07 of 17.10 GB on card 0 while
card 1 sat at ~12.7 GB, and a 557,056-byte decode-graph scratch buffer fell back
to HIP managed memory. The managed allocation was a *symptom* of oversubscription,
not the cause.

### Verified fixes

| Defect | Evidence |
|---|---|
| `vocab_size` fell through to `unwrap_or(32000)` | loader now reports `vocab=248320`; encodes token IDs > 248,000 |
| shape errors swallowed by `.or_else(\|_\|)` | `fallback_on_missing` propagates `ShapeMismatch`; 3 unit tests |
| `hipModuleLoad` 209 | standalone HIP probe: same `gfx1201` object → status 0 on device 0, 209 on device 1. Root cause was process-wide `HSA_OVERRIDE_GFX_VERSION=10.3.0` set by probing the **Ryzen 9800X3D iGPU (card 2, `gfx1036`)**. RDNA4 removed from the override table |
| `GRIM_CONTEXT` ignored on GGUF | 232192 → 32768; KV reservation 32.3 GB → 4.6 GB |
| static round-robin layer placement | headroom-based planner; reserves KV before assigning layers |
| KV arena stride 256 vs 1024 | 3-D eager arena read via `dims().last()`; now multiplies trailing dims |

Test results at commit `10e57867` / `69099316`:

```
grim-core            70 passed; 0 failed
qwen35 (unit)         8 passed; 0 failed
paged KV int8         max err 0.0018  (gate 0.15)  PASS
paged KV FP8 E4M3     max err 0.0223  (gate 0.25)  PASS
multi-GPU context     2 passed; 0 failed
```

### KV quantization

`launch_paged_attention_quant` inlines `dequant_kv_element` for Int8, W4A16,
FP8 E4M3/E5M2, FP4 E2M1, and MXFP4/MXFP8. Int8 and FP8 E4M3 are verified on
hardware against an f64 host reference. FP8 E4M3 is the intended RDNA4 default:
same 1 byte/element as int8, but floating-point dynamic range, which matters for
K where a per-tensor int8 scale flattens outlier channels. Note the paged KV read
is a gather, not a matmul, so it uses the software decode either way — FP8's win
here is accuracy-per-byte, not speed.

Both formats must be uploaded with `from_cpu_bytes`: `dequant_kv_element` indexes
the page pointer **bytewise**, so widening int8 values into f32 slots misaligns
every read by 4x. That was a real failure caught by the parity test.

**Nutcracker is already decoded on the KV path.** `KvCacheQuantFormat::NutFp4 = 8`
exists, and `dequant_kv_element`'s `quant_format == 8` arm reads the per-16
inline block scale (`[exp:6 | sel:2]`, `scale = 2^(exp-31)`) and the repurposed
zero encoding that emits the block's special value, with the sign taken from the
selector rather than the nibble. It is the one variant that ignores the caller's
per-tensor `k_scale`/`v_scale`, because the scale travels in the buffer. The
weight path also has `launch_dequant_nutcracker`, `launch_nutcracker_gemv`,
`launch_nutcracker_gemm_tiled`, and `QuantMode::NutFp4Emulated`.

All three now have parity gates against an f64 host reference:

| format | bits/elem | gate | max err | result |
|---|---|---|---|---|
| int8 | 8 | 0.15 | 0.0018 | PASS |
| FP8 E4M3 | 8 | 0.25 | 0.0223 | PASS |
| Nutcracker | ~4.5 | 1.0 | 0.0761 | PASS |

Nutcracker's gate is much looser because 4-bit E2M1 codes sit on a coarse grid;
it still catches O(1) breakage (wrong block indexing, ignoring the inline scale,
misreading the stolen selector). It uses the production
`grim_quant::quant_nutcracker` encoder, so the test measures the device decoder
against the real encoder rather than a mirror of it. At ~4.5 bits/element it is
the format that would actually make a long context fit.

### Still open

- Qwen35 decode is not routed onto the paged quantized path, so no KV memory
  saving is realized yet.
- The f32↔f16 KV cast was implemented and **reverted**: it compiled, but the
  `RocmDevice` override was never dispatched from the integration test, so the
  conversion is unverified. Deferred.
- Packed KQuant KV is deliberately absent from the dense decode-graph arena; that
  arena's consumer is the non-quantized attention kernel, which reads rows raw.


LFM2 Q8_0 does not need a forced-upload change: GRIM's `WeightSource` already
keeps KQuant/FloatPack bytes packed and device-resident on ROCm, and
`Linear::forward` dispatches them through `quantized_matmul`. The Qwen failure
is specific to the layer-split/Q4 scalar-kernel path, not a general LFM upload
regression.

## int4 KV graph kernel pair (2026-10-15)

The int4 dense KV arena is now served by a capturable kernel pair
(`grim_kv_append_i4` + `grim_qkv_attention_dev_i4`), so int4 runs the decode
GRAPH instead of falling back to eager. Before this, int4 reached capture and
silently ran the int8 kernels (the live hazard the format-derived
`KvArenaFormat::graph_kernel_pair()` predicate now closes).

### Decode throughput — int4 vs int8 vs f16 (T4.1)

Command (release build, single GPU so the shared multi-GPU P2P path is not in
the measurement; 339-token prefill, 47 decode tokens, ctx 4096):

```
HIP_VISIBLE_DEVICES=0 GRIM_CONTEXT=4096 GRIM_KV_ARENA=<arm> \
  target/release/grim-cli run models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf "<wikitext prompt>" \
  --max-tokens 48 --raw
```

| arena | KV arena | ms/token | tokens/sec |
|---|---|---|---|
| f16  | 134 MB | 81.72 | 12 |
| int8 |  68 MB | 80.25 | 12 |
| int4 |  35 MB | 80.70 | 12 |

**Finding: int4 is NOT faster than int8 (or f16) at decode here.** All three
arms land within ~2%, so batch-1 decode at this scale is not KV-read-bound —
the per-step weight GEMV dominates and the KV bytes are a small fraction of the
traffic. int4's justification is KV MEMORY (35 MB vs 68 MB vs 134 MB, a 2x/4x
reduction that buys context length), not decode speed. Reported rather than
shipped as a speed win.

### Long-context retrieval (T4.2)

`crates/grim-cli/tests/kv_int4_retrieval.rs` plants a unique passphrase at 10%,
50% and 90% depth of a ~1200-token haystack and compares the LOGPROB of the
correct answer token under f16 and int4 (a single-bit answer difference is a
coin flip at 4 bits, so the logprob margin is the signal).

| depth | f16 logp | int4 logp | delta | int4 rank of correct token |
|---|---|---|---|---|
| 0.10 | -0.446 | -0.429 | +0.016 | 0 (argmax) |
| 0.50 | -0.565 | -0.630 | -0.066 | 0 (argmax) |
| 0.90 | -0.791 | -1.335 | -0.543 | 0 (argmax) |

int4 retains every needle (correct token is the argmax at all depths). The
logprob penalty grows with depth (up to 0.54 nats at 90%), which is a known
characteristic to watch, not a gate failure (margin 2.0 nats).

### PPL gate (T4.3)

`GRIM_PPL_GATE_ARENA=int4` on Qwen3.5-9B-Q4_K_M passes: rel_delta 0.0136 against
the f32 arena at ctx 4096, within the `grim_quant::accuracy_gate` budget. The
eager path is undisturbed.

## Multi-GPU decode layout (2026-10-16)

### The graph cannot span GPUs

`dc75229d` removed the single-device guard from the Qwen35 decode graph and
captured a layer-split model (16 layers on `rocm:0`, 16 + head on `rocm:1`)
into ONE HIP graph, relying on `enable_peer_access`. Every arena format then
faulted on the first replay:

```
Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
```

f16 included, and f16 uses no packed KV kernel — so this was never the KV
representation.

**Root cause:** a HIP graph is recorded on ONE device's stream, and ROCm cannot
capture a kernel launched on a DIFFERENT device into it. The launch is accepted
during capture; the node faults on replay. Isolated reproducer (no model
needed):

```
cargo run -p grim-backend-rocm --example p2p_graph_probe
# one add_into on device 1 captured while device 0's stream captures
# => Memory access fault by GPU node-1 ... Page not present
```

Peer access does NOT help: `enable_peer_access(0,1)` and `(1,0)` both succeed on
this box (Pcie class, verified with the peer probe) and the fault still occurs.
The failure is the capture/replay DEVICE MISMATCH, not pointer reachability.

**Fix (`75cb675e`):** restore the single-device guard in
`Qwen35::get_or_create_decode_graph`. A layer-split model uses eager, which
moves the hidden state per block explicitly. Qwen35 is the only model that can
be layer-split (sole caller of `plan_layer_devices`).

### The split is a net LOSS for a model that fits on one card

f16, 339-token prefill, 23 decode tokens, ctx 4096, same prompt, quiet box:

| layout | path | ms/token |
|---|---|---|
| single GPU | graph | **89.02** |
| single GPU | eager | 145.22 |
| 2 GPUs (16+16) | eager (graph refused) | 198.42 |

Splitting across two cards is **2.2x slower** than leaving the model on one card
with the graph. `resolve_discrete_rocm_devices` hands the planner EVERY discrete
GPU, and `plan_layer_devices` splits whenever more than one is present — so a
9B that fits in one 17 GB card is split anyway and loses the graph. The split is
only worth its per-token cross-device handoff (`move_to_device` = D2H + H2D +
device drain) when the model does not fit on one device.

Recommendation: split only when the weights + KV arenas do not fit in the
largest single device's budget. Making the graph work on a split model needs ONE
GRAPH PER DEVICE with the boundary hidden state handed off between replays.

## Per-device graphs on the 27B layer split (2026-10-16)

Phase A of `plans/PLAN-multigpu-per-device-graphs.md` landed: the single-device
refusal is gone and a layer-split model captures ONE GRAPH PER DEVICE with an
ordered peer handoff (A2–A6). Split parity gate: worst 2.4e-5 vs eager over 3
steps (bar 1e-1). First production evidence, Swift-1.5-Qwen3.8-27B GSQ-RCO
(IQ3_XXS + IQ1_S banks), both cards, `capacity-aware placement: rocm:0=29
rocm:1=35`, `GRIM_CONTEXT=4096 GRIM_KV_ARENA=int8`, prompt "Paris, and the":

| path | ms/token | tokens | note |
|---|---|---|---|
| **graph (per-device)** | **4395.7** | 31 | `decode-graph: active` |
| eager (graph disabled) | 14688.6 | 16 | `GRIM_DECODE_GRAPH=0` |

**3.34x** — the launch-overhead win compounds across 64 layers x ~10 ops on a
split where every layer boundary previously paid full eager dispatch. The
absolute number is dominated by the IQ1_S scalar fused-dequant GEMV (2.4 GB/s
achieved, weight-bound), which is the next optimization target (D-phase), not a
graph defect. To make the graph capturable at all, the graph decode dispatch
gained: IQ2S embedding gather + IQ1S/IQ1M/IQ2S/IQ2XS/IQ2XXS/IQ3XXS fused-dequant
arms in `linear_decode_into` (outer or-pattern + inner match; the aliased
IQ1S-on-Q2S layout was the historic 27B page fault and must never return).

## D2-2: group-hoisted IQ fused-dequant GEMMs (2026-10-17)

The scalar fused-dequant GEMM called the per-weight dequant for every (output,
weight) pair — block header, scale/index math and grid-table load EVERY weight.
rocprofv3 on the shape-exact probe: grid 17408x256, no LDS, ~5 ms for 17.4 MB =
~1% of card peak, i.e. ALU/latency bound. Within an 8-weight group all of it is
constant, so the grouped kernels hoist it to the group.

Swift-1.5-Qwen3.8-27B GSQ-RCO, layer split (29/35), `GRIM_CONTEXT=4096
GRIM_KV_ARENA=int8`, prompt "Paris, and the", 32 tokens, both cards:

| stage | ms/token | GEMV GB/s |
|---|---|---|
| eager (pre-Phase-A baseline) | 14688.6 | 0.7 |
| per-device graph, scalar IQ kernels | 4395.7 | 2.4 |
| + IQ1S/IQ3XXS/IQ3S/IQ2S/IQ2XS/IQ2XXS grouped | 626.3 | 16.7 |
| + IQ4_XS/IQ1_M grouped (all eight) | **606.3** | 17.2 |

**24.2x vs the eager baseline**; the IQ1_S probe alone went 4.07 -> 113.1 GB/s
(27.8x). All eight formats are BIT-IDENTICAL to their retained scalar oracles
(worst abs 0.000e0 at n=256/k=5120) — same values, same ascending accumulation
order, so no PPL revalidation is needed for the format change. Regression: qwen
6/6, lfm2 11/11, backend lib 594/594.

## Single-device fit: do not split what fits (2026-10-17)

`assign_by_headroom` splits proportionally to headroom by design and never asks
whether one device suffices. At ctx 4096 / int8 KV both target models fit the
primary card, so the split was pure loss — a cross-device hop per forward:

| model | split (2 seg) | single device | delta |
|---|---|---|---|
| Qwen3.5-9B-Q4_K_M | 117.63 ms/token | **83.23** | -29% |
| 27B GSQ-RCO IQ3_XXS | 606.3 | **430.2** | -29% |

Fit is checked against the PRIMARY device's remaining budget after its KV arena
share (9B 4.26/14.03 GB; 27B 9.05/9.63 GB). `GRIM_FORCE_SPLIT=1` opts out so the
multi-GPU path stays testable — the split gates set it. The split remains the
fallback for models that do not fit; its per-device graph path is green
(parity 2.4e-5) and 3.34x over eager when it is used.

## Prompt processing: WMMA removed, and the reuse gap (2026-10-17)

`launch_iq_wmma_fallback` routed `m <= 4` to the fused WMMA kernel — the SHORT
PROMPT through the path it is worst at. Measured on the 27B (single device, ctx
4096, int8 KV): the WMMA fused-dequant kernels were 19.6 s of a run's 21.6 s of
kernel time at ~40 ms per tensor (~2.5 GFLOP/s). The group-hoisted scalar
kernels do the same shapes in ~0.5 ms, and a 110-token prompt confirms it is not
a small-m artifact: scalar 13.7 s wall vs WMMA 86.8 s (6.3x). WMMA is now
never preferred for the IQ formats (`GRIM_WMMA_M_MAX` restores it for A/B).

| 27B, 4-token prompt, 32 tokens | wall | decode |
|---|---|---|
| WMMA at m<=4 (old) | 28.3 s | 424.2 ms/token |
| scalar (now) | **9.0 s** | **136.0 ms/token** |

Token sweep after: n=8 5.9 s, n=32 8.4 s, n=64 11.9 s → startup 18.5 → **~4.9 s**,
steady decode 172 → **109 ms/token** (135x the 14688.6 ms/token eager baseline).
Output tokens unchanged.

**Remaining prefill gap, measured.** Prompt processing is LINEAR in prompt
length at ~63 ms/token (40 tok 2.5 s, 120 tok 8.2 s, 360 tok 21.8 s) because the
grouped kernel has no reuse across prompt rows: every output element re-reads
its weight row. Reading the 9 GB weight set ONCE at ~300 GB/s would cost ~30 ms
for the whole prompt regardless of length. Closing that needs a tiled IQ kernel
that stages a dequantized weight tile in LDS and reuses it across TILE_M prompt
rows — the same structure the K-quant tiled kernels already use, extended to the
IQ grid formats (iq1s/iq1m have no tiled stamps at all). Not implemented; this
is the measured target.

## Prompt tiling: implemented, gated, measured NEUTRAL (2026-10-17, 51536ccd)

Prompt processing is linear in prompt length at ~63 ms/token, which looked like
weight re-reads (the grouped kernel dequantizes each weight row once per output
element). `grim_fused_dequant_gemm_iq1s_tiled` stages the dequantized weights in
LDS once per super-block and reuses them across TILE_M=16 prompt rows — 16x less
weight traffic and dequant work. Bit-identical to the grouped kernel (parity
gate: worst abs 0.000e0 at m=16/33/40 including a partial final tile and a
non-multiple n).

Measured at the production shape (n=17408, k=5120), grouped vs tiled:

| m | grouped | tiled | ratio |
|---|---|---|---|
| 110 | 21.7 ms | 21.7 ms | 1.00x |
| 339 | 58.5 ms | 60.0 ms | 0.97x |

**No win.** The kernel is COMPUTE-bound — scalar dequant + FMA at ~500 GFLOP/s —
not weight-traffic-bound, so removing 16x of the weight traffic buys nothing.
The linear growth in m is m*n*k FLOPs, not the re-read. Tiling is default OFF
(`GRIM_IQ_TILED_M_MIN=<m>` opts in); the kernel and its gate stay as the record
of the measurement and as the base for real vectorization, which changes the
accumulation order and therefore needs a tolerance/PPL gate rather than 0.0
parity. **The remaining prompt lever is the per-FLOP rate, not reuse.**
