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

### Follow-up: the per-FLOP lever, measured (2026-10-17, a5647d43)

After tiling measured neutral, the next hypothesis was load issue: the grouped
kernel's inner loop is one FMA per activation load. A bit-identical vec4 variant
(float4 activation loads, unchanged accumulation order, so the 0.0 parity gate
covered it) measured **52.0 vs 27.9 ms at m=110 (1.9x SLOWER)** and slower at
m=1 — the eight grid byte extractions cost more than the load issue they saved.
Removed rather than shipped.

Two measured negatives on this front (tiling 1.00x, vec4 0.52x) are the evidence
that the IQ GEMV is near its practical limit in this kernel structure. Further
gains are structural — matrix cores with a vectorized dequant, or a q8_1-act
integer dot — and both change the accumulation order, so they need a
tolerance/PPL gate rather than bit-parity.

## Task-graph ring readiness by model (Phase M, 2026-10-17)

Phase M emits a per-step task graph during capture and asks, per model, how much
of it a persistent device-side ring (M4 onward) can execute. The predicate
`task_graph::ring_ready(&CapabilitySet)` is DERIVED from the emitted IR — never
a hand-maintained model list — so a model that gains an opcode is re-classified
automatically, and a `full: false` verdict always names what is missing.

Measured with `GRIM_GPU_TEST=1 GRIM_CONTEXT=512`, each checkpoint loaded through
its OWN model type (`cargo test -p grim-cli --test model_zoo_ir_coverage`; never
routed through `Qwen35::load`). Re-measured 2026-10-19, after M5:

| Model | Shape | IR nodes | Opcodes in the IR | `ring_ready` |
|---|---|---|---|---|
| `models/Llama-3.2-1B-Instruct-Q4_1.gguf` | dense attention | 35 | GEMM(1), ATTENTION(8), NORM(10), ELEMENTWISE(12) | **`full: true`**, `missing=[]` |
| `models/Xing4.0/Xing4.0-29B-A4B-IQ3_M-MTP.gguf` | MoE + MLA | 83 | GEMM(1), MOE_DISPATCH(6), ATTENTION(8), NORM(10), ELEMENTWISE(12) | **`full: true`**, `missing=[]` |

`RING_OPCODES` names the opcodes the ring implements. M4 converted **op::GEMM**;
M5 added NORM, ELEMENTWISE, ATTENTION and MOE_DISPATCH — which is every opcode
either zoo model emits — so both now report `full: true`:

| Model | `missing` after M5 | Still unconverted |
|---|---|---|
| llama-dense | (none) | 0 |
| xing40-moe-mla | (none) | 0 |

`Q8_0_DOT4` is deliberately NOT listed: the wave has no dot4 arm, and the number
it was given (8) is the wave's `OP_PEER_REDUCE` — so listing it made `ring_ready`
report coverage that did not exist, and binding it would have dispatched an
all-reduce for a quantized GEMM. It returns when the arm does, which is the
plan's PM-6 target. `every_advertised_ring_opcode_maps_to_an_implemented_wave_arm`
parses the kernel's own `#define OP_*` lines and fails if any advertised opcode
has no arm.

The gate derives BOTH `full` and `missing` from `RING_OPCODES` rather than
asserting "everything is missing" or "never full", so a constant predicate cannot
pass and the table above is a consequence of the code, not a restatement of it.
Measured by `cargo test -p grim-cli --test model_zoo_ir_coverage`.

The conversion itself: M4's ring binds the executable GEMM subset of an emitted
graph (role-resolved operands, real geometry, native weight only), stages the
descriptors once, and drains them with the existing persistent wave. On the real
Llama checkpoint 16 of 35 IR nodes convert and the ring's logits are
**bit-identical** to the launcher path (`0.000e0`, three steps). The wave's GEMM
arm is the existing scalar f32 reference, so a QUANTIZED GEMM still runs on its
launcher — a Q8_0 dot4 arm is the next entry. One launch per STEP (rather than
per node) needs a resident wave on its own stream; that is task M6, because a
non-resident wave on the caller's stream serializes with the host's attention
kernels.

### Per-node quantization coverage (the GEMM axis)

`params[0]` on a GEMM node carries that weight's `QuantFormat` id (`0` = native,
host-dequantized at load), so the axis is measured rather than asserted:

| Model | GEMM `params[0]` | Reading |
|---|---|---|
| Llama-3.2-1B Q4_1 | `[0, 0, …, 0, 7]` | sixteen host-dequantized Q4_1 linears (`0`) plus a packed **Q6_K** head (`7`) |
| Xing4.0-29B A4B | `[5, 5, 7]` | two dense lead blocks in **Q4_K** (`5`) plus a **Q6_K** head (`7`) |

Q4_0/Q4_1 have no `QuantFormat` and no device dispatch, so they are
host-dequantized by design — the Q4_1 row is exercised as *decoder* coverage,
not as an axis value. Xing4.0's 38 MoE layers carry their expert counts in
`params[0..2]`, so MoE banks are not yet on the axis (recorded gap). The head
GEMM's `params[1]` carries the output width (the logits length), kept when the
axis moved to slot 0.

### What the IR must not lie about (2026-10-17)

The Xing40 emitter chained both its final NORM and its head GEMM off the last
layer's node, giving the IR **two sinks** and telling a scheduler it could run
the head before the norm. The PM-4 gate caught it on re-run — a reminder that an
emitter and the graph it publishes are two separate claims, and only the gate
tests the second.

## The M=1 decode regime: three experiments, then the kernel they licensed (2026-10-22)

Plan (local working document, alongside the sibling
`plans/META-SYNTHESIS-res1-res8-vs-rocm-kernels.md` that named the three
experiments): `plans/PLAN-m1-decode-regime-experiments.md`. Corpus:
`old/res8/` (76 papers, 2026). The three experiments are the ones
`plans/META-SYNTHESIS-res1-res8-vs-rocm-kernels.md` §4 #5 named as "nobody
has run".

Card: **RX 9060 XT (gfx1200), ordinal 1** — the measurement card. Card 0
(RX 9070 XT, gfx1201) is reserved and was not used. ROCm 7.2.4, wave32.

### (b) rho = TC:CC — DONE

`tests/rdna4_rho_microbench.rs` (the infrastructure already existed; there
was no recorded result).

```
[rho] fp32 VALU: 6.4 TFLOP/s | wmma f16 TC: 70.1 TFLOP/s | rho = 11.04
```

**rho = 11.04 >> 1**, so the "the CUDA cores are too slow to feed the
tensor cores" branch is decisively rejected: dequant work on the VALU is
cheap next to the TC, and a fused-dequant matrix-core path is structurally
right for M>1. Two consequences the plan did not anticipate:

- **The IQ family cannot use the tensor core at all.** Their 4-bit codes
  are table *indices*, not numbers, and gfx1200/1201 expose no int8 WMMA
  (the int8 matrix path arrives with gfx950). The WMMA route can serve the
  K-quants and the u4-lane path; it can never serve the IQ family, whose
  improvement must come from the *lookup*.
- **For M=1 rho is nearly irrelevant** — there is no MMA at M=1 — so it
  does not move the decode plan.

Caveat: the plan asked for gfx1201; the measurement is gfx1200 because
card 0 is reserved. Both are RDNA4 and both CC and TC scale with CU
count, so rho is expected to carry over; the expectation is **rho ~ 11,
not a measured 11.04 on gfx1201**.

### (a) The compute-only decode GEMV — DONE, and it moves the diagnosis

New probe: `crates/grim-backend-rocm/tests/m1_decode_regime_probe.rs`
(working-set sweep, cold/warm pair, and a **kernel census**).

**The census is not optional.** The first sweep set `GRIM_DOT_GEMV=1` and
measured "Q4_K", and Q4_K came out 12x faster than the committed 81.2
GB/s. The trace explains it: **the Q4_K arm of `linear_decode_into` takes
the u4-lane route** — a LOSSY W4A4 requantization — because
`try_u4_lane_decode` allows it for Q4_K and refuses it for Q5_K/Q6_K on
fidelity grounds. The first sweep measured a different weight
representation, not a different unpack.

| format | kernel actually launched | per launch, n=4096 k=5120 m=1 | VGPR | scratch spill |
|---|---|---|---|---|
| Q4_K | `grim_quantize_u4_group128` + `grim_dot8_w4a4_gemv` | 29 us (u4-lane) | - | - |
| Q5_K | `grim_dot4_q5k_q81_gemv` | 245 us | **192** | **180 B** |
| Q6_K | `grim_dot4_q6k_q81_gemv` | 178 us | **192** | **88 B** |
| Q2_K | `grim_dot4_q2k_q81_gemv` | 120 us | 64 | 0 |
| Q3_K | `grim_dot4_q3k_q81_gemv` | 145 us | 88 | 0 |
| Q4_K (exact) | `grim_dot4_q4k_q81_gemv_fast` | 130 us | 72 | 0 |
| Q8_0 | `grim_dot4_q80_f32act_gemv` | 55 us | 64 | 0 |

**Working-set sweep** (k=5120, m=1, working set 3 MB -> 138 MB; `t_cold`
rotates over 8 weight copies so the bytes come from DRAM):

| format | marginal GB/s | t_cold/t_warm at the 27B shape | fixed per-launch cost |
|---|---|---|---|
| Q4_K (exact) | 98 | 0.90 | ~0-10 us |
| Q6_K | 85 | 0.97 | ~0 us |
| Q2_K | 78 | ~1.0 | ~0 us |
| Q3_K | 62 | - | - |
| Q5_K | 59 | 0.97 | ~9-12 us |
| Q8_0 | 39 | 1.00 | ~20 us |
| Q4_K u4-lane (lossy) | **~300 warm** | 0.90 | - |

**And the three-way split, measured rather than inferred.** The plan asked for
three tests. Two were done as measurements and the third turns out to need no
separate build:

1. **BANDWIDTH — not constrained.** Cold vs warm at matched shape (8 weight
   copies rotating, so the cold arm's bytes come from DRAM): `t_cold/t_warm` is
   **0.90-1.00** at every format and every working set from 3 MB to 138 MB.
2. **LAUNCH — not constrained at the dominant shape, constrained at small
   ones.** Measured directly: the kernel's own fixed cost (launch + prologue +
   shuffle reduction) is **33.5 us** at the minimal shape, not the 0-20 us the
   working-set fit implied. That is **4.9%** of the Q5_K step at n=17408 but
   **27%** at n=1024. The working-set fit's intercept is not the launch term —
   it does not reproduce the measured points at the small end (70 us predicted
   against 122 us measured at n=1024).
3. **COMPUTE-ONLY — the strict variant is unnecessary, and the reason is
   arithmetic.** The dot4 mapping gives one wave 4 output columns, so a wave's
   weight working set is `4 x (K/256) x block_bytes` = **6.6 KB (Q2_K) to
   16.4 KB (Q6_K)** — and gfx1200's L1 is **32 KB per CU**. **Every K-quant
   dot4 wave's weight working set fits in L1**, so the warm arm of the
   cold/warm pair is already the compute-only floor, and the pair shows that
   removing the DRAM term changes nothing. A register-resident variant could
   only raise the ceiling, not explain the current behaviour.

**Three measured conclusions:**

1. **NOT bandwidth-bound.** `t_cold ≈ t_warm` (0.90-1.00) at every
   format and every working set. The DRAM term is fully hidden under the
   kernel's own work.
2. **NOT a per-launch floor.** The fit gives a fixed per-launch cost of
   0-20 us, not the 1-3 ms the floor probe reported for the PRE-HOIST
   kernel at k=4096.
3. **WORK-bound** — the unpack, exactly as `old/res8` §2.1 argues. The
   byte-count counter-example is inside the measurement: **Q2_K reads the
   FEWEST bytes of any K-quant (6.9 MB) and is among the SLOWEST (120 us)**;
   Q6_K reads 2.5x more and is FASTER (178 us).

**The ceiling is not the silicon.** The u4-lane path reaches ~300 GB/s
warm on the same shapes with no per-element dequant at all. The exact
K-quant dot4 legs sit **3-5x below what a different (lossy) weight
representation achieves on the same shape** — and that gap is an
unpack/structure gap, not a bytes gap.

### The kernel change the experiments licensed: Q2_K word-wide-mask unpack

`grim_dot4_q2k_q81_gemv` fetched the four 2-bit codes of a 4-weight group
with **FOUR SEPARATE 1-BYTE LOADS**. Q2_K stores one code per BYTE, so the
four codes are four CONSECUTIVE bytes and **one 32-bit load delivers all
of them**. The mask is exact: the codes are the 2-bit field at bit `shift`
of each byte, `shift = 2*run` in {0,2,4,6}, and after `w >> shift` each
byte's low 2 bits still come from its own byte (the bits shifted in from
the next byte land at bit `8-shift >= 2` and are masked off), so

    (w >> shift) & 0x03030303  ==  the stock per-byte extraction

bit for bit. **The same mask the landed group hoist already uses for the
Q5_K `qh` field.** The dword is 4-byte aligned by construction and the
load is a 4-byte `__builtin_memcpy`, so it is alignment-safe regardless.

**Gate** (`tests/dot4_kquant_grouped_parity.rs`, the `q2k` row): byte for
byte against the stock kernel at five shapes, **both driven through
launchers** so the gate sees the dispatch's grid (the Q4_K lesson: a gate
that called the kernel with its own grid stayed green while the launcher
fed it half the columns), and both against the llama.cpp-faithful
`dequant_q2k` dotted with the same q8_1 codes.

| | per launch | GB/s |
|---|---|---|
| stock (4 byte loads per group) | 0.459 ms | 63.7 |
| **word-wide-mask (1 dword per group)** | **0.374 ms** | **78.2** |
| | **1.23x** | byte-identical (15/15 shapes) |

`GRIM_Q2K_DOT4_FAST=0` keeps the stock kernel reachable.

### The next change, named by the measurement: the two SPILLING legs

**The two slowest exact legs are the two that spill.** `dot4_q5k_q81_gemv`
and `dot4_q6k_q81_gemv` compile to 192 VGPRs with 180 B and 88 B of
scratch spill per thread; the Q4_K `_fast` sibling is at 72 VGPRs and 0 B.
They fully unroll 4 columns x 8 sub-blocks x 8 e-iterations, and the
spill puts a local-memory round trip inside a kernel that is **not**
bandwidth-bound.

This is `old/res8`'s 2607.01590 finding, measured on grim's own silicon:
**a kernel that compiles and passes correctness but silently violates an
on-chip capacity limit runs 20-41x slower than the hardware allows.** The
register file is an on-chip capacity limit. The fix under way is to bound
the unroll on the two innermost loops of each kernel, which is
**bit-identical by construction** (the int32 accumulation is an exact
integer chain whose order does not change, and the float accumulation still
runs once per sub-block in the same order).

### (c) The phase-level breakdown of the decode window — the constant is NAMED

`GRIM_PHASE_TRACE=1` in `run.rs` prints the per-step distribution, the
excess over the steady state, the slowest steps with the path that carried
them, and the per-path totals.

RX 9060 XT (gfx1200), `GRIM_CONTEXT=4096 GRIM_KV_ARENA=int8`, prompt
"Paris, and the", the same model at TWO token counts — the plan's
falsifier ("the constant is identified by what does *not* scale with
tokens"):

| | 8 tokens | 32 tokens |
|---|---|---|
| window | 4.994 s | 12.807 s |
| p50 (the steady state) | **361.7 ms** | **353.6 ms** |
| step_sum | 4.936 s | 12.763 s |
| steady (median x n) | 2.894 s | 11.317 s |
| **EXCESS** | **2.042 s (40.9% of the window)** | **1.447 s (11.3%)** |
| slowest: prefill | 1711.0 ms | 1101.0 ms |
| 2nd slowest: **eager (the capture step)** | **1044.0 ms** | **1022.1 ms** |
| graph-sampled (the steady state) | 6 x 363.4 ms | 30 x 354.7 ms |
| reported ms/token | 460.66 | 376.20 |

1. **The steady state is 354-362 ms/step** and is the SAME at both token
   counts — it is the per-token work.
2. **The one-time cost is the prefill plus the graph capture's eager step,
   and it does not scale with tokens**: the eager step is **1044.0 ms at 8
   tokens and 1022.1 ms at 32** — constant to within 2%. The plan's
   hypothesis (capture running the model eagerly, or the prefill tail) is
   confirmed, and the alternative it had to exclude (a cost that scales
   with the prompt length) is excluded for the eager step.
3. **So a short benchmark run overstates ms/token by up to 25-40%.** At 8
   tokens the reported 460.66 ms/token is 361.7 ms of per-token work plus
   2.042 s of one-time cost smeared over 8 tokens; at 32 tokens the
   reported 376.20 ms/token is 353.6 ms plus 1.447 s. **Every ms/token
   figure in the corpus measured over a short token count carries this
   inflation** — including the plan's own "606.3 ms/token" for the 27B
   (32 tokens), whose ~12 s constant is this same structure scaled by the
   model.

**What this kills.** The plan had already ranked the host-side per-token
work (logits D2H, sampling, MTP bookkeeping) LAST, on the grounds that
the unexplained time is a constant rather than per-token work. That is now
measured rather than argued: the per-token host work is inside the
354-362 ms steady state, and the one-time cost is prefill + capture.

**The actionable consequence.** The capture's eager step is a **one-time
1.02 s on the 9B**, paid on the FIRST decode step (where the u4-lane
prewarm also lands). A benchmark that reports an average over a short run
is mostly reporting its own startup. Report the **median step** and the
**excess** separately, and treat the capture step as a startup cost to be
amortised over a long run.

### The two kernel changes the experiments licensed, and their measurements

Both are gated on **byte-identity** against the retained oracle
(`tests/dot4_kquant_grouped_parity.rs`), both driven through **launchers**
so the gate sees the dispatch's grid — the Q4_K lesson: a gate that called
the kernel with its own grid stayed green while the launcher fed it half
the columns — and both checked against the llama.cpp-faithful
`dequant_*` dotted with the same q8_1 codes, so byte-identity cannot be
satisfied by two identically wrong kernels. **15/15 shapes byte-identical.**

| | per launch, n=17408 k=5120 m=1 | GB/s | change |
|---|---|---|---|
| **Q5_K** | 1.090 -> **0.687 ms** | 56.2 -> **89.2** | **1.59x** |
| **Q6_K** | 0.923 -> **0.732 ms** | 79.2 -> **99.9** | **1.26x** |
| **Q2_K** | 0.459 -> **0.372 ms** | 63.7 -> **78.5** | **1.23x** |
| Q4_K | 0.606 -> 0.614 ms | 82.7 -> 81.7 | unchanged |

**Change 1 — the Q2_K word-wide-mask code unpack.** The stock
`grim_dot4_q2k_q81_gemv` fetched the four 2-bit codes of a 4-weight group
with **FOUR SEPARATE 1-BYTE LOADS** and re-assembled them with four
shifts and three ORs. Q2_K stores one code per BYTE (84 B / 256 weights,
`qs[64]` at offset 16), so the four codes of a group are four CONSECUTIVE
bytes and **one 32-bit load delivers all of them**. The mask is exact: the
codes are the 2-bit field at bit `shift` of each byte, `shift = 2*run` in
{0,2,4,6}, and after `w >> shift` each byte's low 2 bits still come from
its own byte (the bits shifted in from the next byte land at bit
`8-shift >= 2` and are masked off), so

    (w >> shift) & 0x03030303  ==  the stock per-byte extraction

bit for bit. **The same mask the landed group hoist already uses for the
Q5_K `qh` field.** The dword is 4-byte aligned by construction and the
load is a 4-byte `__builtin_memcpy`, so it is alignment-safe regardless.
`GRIM_Q2K_DOT4_FAST=0` keeps the stock kernel reachable.

**Change 2 — the unroll bound on the two SPILLING legs.** The Q5_K and
Q6_K kernels fully unrolled 4 columns x 8 sub-blocks x 8 e-iterations and
compiled to **192 VGPRs with 180 B and 88 B of scratch spill per
thread** — the same shape as the Q4_K `_fast` variant's own history ("the
stock 4-col kernel compiles to 192 VGPRs + 560 B scratch -> 1 wave/SIMD
plus local-memory spills; the 88-VGPR q8_0 sibling sustains 300 GB/s").
The spill puts a local-memory round trip inside a kernel that is **not**
bandwidth-bound. Bounding the unroll on the two innermost loops is
**bit-identical by construction**: the int32 accumulation is an exact
integer chain whose order does not change, and the float accumulation
still runs once per sub-block in the same order.

| kernel | VGPR | scratch | |
|---|---|---|---|
| `dot4_q5k_q81_gemv` | 192 -> **64** | 180 B -> **0** | |
| `dot4_q6k_q81_gemv` | 192 -> **104** | 88 B -> **0** | |
| `dot4_q4k_q81_gemv_fast` | 72 | 0 | |
| `dot4_q3k_q81_gemv` | 88 | 0 | |
| `dot4_q2k_q81_gemv_fast` | 96 | 0 | |
| `dot4_q80_f32act_gemv` | 64 | 0 | |

**No K-quant dot4 kernel spills any more** — every one is in the same
64-104 VGPR band as the Q8_0 sibling that sustains 300 GB/s. The two legs
that spilled are the two that moved most.

**A trap worth recording.** The first attempt at change 2 applied the
unroll bound to the **Q4_K stock kernel** instead of Q5_K: the two share
the same `for (int is = 0; is < 8; ++is)` loop text, and a first-match
text replacement does not know which kernel it is in. The kernel census
caught it (Q6_K's VGPRs moved and Q5_K's did not). The edit is now scoped
to the Q5_K kernel's own region.

### End to end: what the kernel win is worth, and what it is NOT

The same 9B run, same card, same prompt, 32 tokens, before and after the
two kernel changes (`GRIM_PHASE_TRACE=1`):

| | p50 (the steady state) | reported ms/token |
|---|---|---|
| before | 353.6 ms | 376.20 |
| **after** | **348.4 ms** | 370.68 |

**1.5% end to end, and that is the honest number.** The kernel-level win
is 1.56x on Q5_K and 1.28x on Q6_K, but the 9B's Q4_K_M mix is
**Q4_K 104, Q5_K 131, Q6_K 30, Q3_K 7, IQ4_XS 117, Q8_0 106** — the
K-quants are a minority of the decode work, and the two families that
dominate the count (**IQ4_XS at 117 and Q8_0 at 106**) are untouched by
these changes. A 1.56x on the K-quant projections cannot move a
mixed-format decode far, and the measurement says so rather than the
headline.

**This is where the corpus's mechanism for the IQ family becomes the next
lever**, and it is a bigger prize than the K-quants: 117 of the 9B's
tensors and 117 of the 27B's are IQ4_XS, and that path runs at **17.2
GB/s** against the K-quants' 78-100.

### What is left

- **Q3_K (62.1 GB/s, 88 VGPRs, no spill)** is the last K-quant format
  with a byte-wise unpack. Its `qs` is one 2-bit code per BYTE, so the
  same `0x03030303` mask applies to its magnitude plane; its sign plane
  is a separate bit-plane that cannot be masked, so the win will be
  smaller than Q2_K's.
- **The IQ family** (17.2 GB/s GEMV on the 27B; IQ4_XS is 117 of the
  Q4_K_M mix's tensors) is the largest remaining low-bandwidth path, and
  it is the one the corpus has a mechanism for that the tree does not:
  **CodeGEMM's Psumbook** (precompute each centroid's inner product
  against the activation; the hot loop becomes a code-indexed gather of
  FP16 partial sums; ~0.9% build overhead; 1.83x over AQLM) and the ETH
  paper's one-stage argument (the two-stage "build a LUT then accumulate"
  pipeline is what makes the IQ-style kernels slow; 1.0-18.1x over
  RSR++). Experiment (b) bounds this: the **tensor core cannot serve the
  IQ family at all** (its 4-bit codes are table indices, not numbers, and
  gfx1200 exposes no int8 WMMA), so the IQ path's lever is the lookup.

## The IQ4_XS decode census: the 17.2 GB/s premise is refuted (2026-10-22, PLAN-iq-family-decode T1)

`iq4xs_decode_kernel_census` (n=4096, k=5120, m=1, 11.14 MB packed, GPU 1, rocprofv3 `--kernel-trace`):

| kernel | grid | VGPR | scratch | n | µs/launch |
|---|---|---|---|---|---|
| `grim_fused_dequant_gemm_iq4xs_grouped` | 4096 | 80 | 0 | 23 | **102.7** |
| `grim_quantize_q8_1` (activation pre-quant) | 40960 | 16 | 0 | 23 | 3.1 |

Host-timed: **107.2 µs/launch = 103.9 GB/s effective**. The decode leg takes the
`_grouped` (D2-2 group-hoisted) arm — no u4-lane W4A4 route, no dot4 leg exists
for IQ4_XS, no scratch spill. 103.9 GB/s is ABOVE the stock Q4_K (81.2) and Q6_K
(99.9) dot4 levels: **the "IQ family 17.2 GB/s" figure described the pre-group-hoist
fused kernel and no longer describes the decode leg.** The warp-per-column-group
rewrite (T2) proceeds as an A/B only — it must beat 103.9 GB/s by >1.15× to ship
as a default.

## PLAN-iq-family-decode: T2-T7 results (2026-10-22, commits 41bdd6b5, 1b4f57d9)

### T2 - the dot4 mapping does NOT transfer to IQ4_XS (measured-negative)

A q8_1-activation dot4 pair for IQ4_XS (`grim_dot4_iq4xs_q81_gemv` stock +
`_fast` word-nibble) was built and gated: **15/15 shapes byte-identical
stock-vs-fast through launchers, oracle rel err ~2e-6 vs
`dequant_iq4xs`** (the first run caught a real defect: group `ib`'s 16 code
bytes are `qs+ib*16`, not `qs[0..15]`). Speed at n=17408/k=5120/m=1:

| arm | time | GB/s |
|---|---|---|
| dot4 fast (word nibble) | 3.037 ms | 15.6 |
| dot4 stock (byte nibble) | 3.045 ms | 15.5 |
| production `_grouped` f32-act | 0.688 ms | **68.8** |

0.14x/0.23x vs `_grouped` at n=4096/17408. The codes are kvalues_iq4nl
TABLE INDICES, so the integer dot pays a 16-entry LDS gather per 4 weights
(1.00x fast-vs-stock: the unpack never mattered — the gather dominates).
**`_grouped` stays the dispatch default**; the dot4 pair lives on as
A/B-only launchers. This is exactly what experiment (b) predicted: IQ
cannot ride the tensor core or the integer dot.

### T3 - IQ2/IQ3 census: headroom is real, but the lever is the grid-table path

`iq_family_decode_census` (n=4096, k=5120, m=1, GPU 1, rocprofv3):

| format | kernel | VGPR | scratch | µs | GB/s (packed) |
|---|---|---|---|---|---|
| IQ3_XXS | `_grouped` | 48 | 0 | 158.7 | 46.5 |
| IQ3_S | `_grouped` | 48 | 0 | 175.5 | 47.6 |
| IQ2_S | `_grouped` | 64 | 0 | 165.6 | 37.4 |

2-3x off the K-quant level — but dot4 is **mathematically inapplicable**:
2-3-bit codes index 32-byte grid vectors, not per-weight scalars, so there
is no i8x4 operand to build. The measured traffic structure: each 32-weight
group reads a 32-byte grid-table row from a 1-8 KB `static const` table
(L2-resident) — real traffic is ~2.6x the packed bytes for IQ3_XXS. The
next lever here is restructuring the grid-table read pattern, not the
thread mapping.

### T4 - Psumbook closed without a build (two independent refutations)

1. Arithmetic: at M=1 a codebook LUT's build cost EQUALS the direct cost —
   each 256-entry grid is consumed ~once per super-block position, so there
   is no amortization. CodeGEMM's 0.9% overhead claim is a large-M result.
2. Measurement: the dot4 pair above IS the "gather instead of dequant" arm
   on this hardware, and it lost 4.4-7.5x. The ETH one-stage argument holds:
   the `_grouped` kernels already are the one-stage dequant+accumulate.

### T5 - Q3_K word-wide magnitude plane: byte-identical, NEUTRAL

`grim_dot4_q3k_q81_gemv_fast`: one 32-bit load + `& 0x03030303` replaces
four 1-byte loads per magnitude group; sign plane stays per-byte. 15/15
shapes byte-identical, oracle ~2e-6. **70.2 vs 69.8 GB/s = 1.01x** — the
sign plane and the 12-byte scale shuffle dominate, as predicted. Default
flipped to fast (never worse); `GRIM_Q3K_DOT4_FAST=0` escape.

### T6 - L1/L2 counter split: BLOCKED, recorded analytically

rocprofv3 `--pmc` replays the whole test once per counter pass; on this
census test it exceeded 12 minutes and was killed (rocprof v1 unsupported
on this GPU). Retry path: a single-launch microtest per kernel, not the
census. Analytical split in place of counters: K-quant dot4 waves are
L1-resident by footprint (6.6-16.4 KB < 32 KB/CU); the IQ grids are
L2-resident by table size (1-8 KB).

### T7 - the long-window protocol, validated

9B (Q4_K_M, GPU 1, ctx 4096, --raw, 72 generated tokens, GRIM_PHASE_TRACE):
window 28.104 s, p50 356.0 ms, p90 362.1, **EXCESS 2.389 s = 8.5%**
(11.3% at 32 tokens, 40.9% at 8), reported avg 366.39 ms/token vs steady
356.0 — short-run inflation shrinks to ~3% at 72 tokens. Output coherent;
decode-graph replay served every step. **Standing rule: report steady p50
with EXCESS separated; a ms/token from a <32-token window is not comparable.**

## Tiered lever implementation: res0/res9/res10 adoption round (2026-10-22)

Commits 8e5de37e, 6ce1a56d, 189e047d, 8c471de6, and the two that follow.
All gates byte-identity through launchers on GPU 1 unless stated.

### SHIPPED

**IQ codebook-cache LDS variants (T1-1, 8e5de37e)** — VQ-LLM/SPHQuant lever.
Tables staged in `__shared__` once per block (cooperative, block-size-agnostic
strided preload); math and order unchanged, so the pair is byte-identical.
iq3xxs **1.81x** n=4096 (91.3 GB/s) / 1.15x n=17408 → DEFAULT
(`GRIM_IQ_LDS=0` escape). iq2s **2.04x** / 1.11x → DEFAULT. iq3s 1.18x but
**0.78x** at n=17408 (preload+syncthreads beat by L2 amortization at large N)
→ stays stock, recorded. Two lessons banked: LDS byte-array rows (`[512][4]`)
measured 0.77x — rows must be word-packed with in-register byte extraction;
LDS byte reads are slow.

**Q4_K DMMV num-rows variant (T1-4, 8c471de6)** — zolotukhin's
chunked-prefill lever. One workgroup = 2 columns x ALL M rows (M<=8);
weights decoded once instead of M times. Byte-identical to M single-row
launches at M=1/2/4/8. A/B: **1.20x (M=2), 1.22x (M=4), 1.32x (M=8)**
n=4096; 1.22x n=17408 M=8. Modest vs the 8x read saving because the
single-row path was already partially cache-resident. Probe launchers only
(`launch_dot4_q4k_q81_gemv_dmmv_for_ab`); needs a chunked-prefill caller to
ship.

### MEASURED — DECISION POINTS

**INT8 WMMA prefill (T1-3, 6ce1a56d)** — rocwmma int8 on gfx12 REJECTS
int8 x uint8 (SOA_INT layouts differ), both operands signed; unsigned weight
codes map via (code-8) + the -8*sum(a) two-dot correction. i32 gate
BIT-IDENTICAL at 3 shapes. Naive 16x32 tile: 1.9-8.3 TFLOP/s (B-re-read
bound); 4-M-tile B-reuse: **12.4 TFLOP/s ceiling** — 10x under grim's
measured f16 prefill. The 2x nameplate I8 rate does not materialize either:
**measured rho (T3-9) = f16 46.9 / I8 63.7 TFLOP/s = 1.36x, not 2x**.
The lever stays closed unless the big-tile skeleton (LDS staging, double
buffering) is ported AND the measured 1.36x rate justifies it — it does not
today.

**Sparsity opportunity (T1-2, 189e047d)** — GRIM_SPARSE_PROBE on the 9B's
real decode activations: dead 32-wide k-groups (tau x global-max):
**70.1% at tau=0.10**, 37.2% at 0.05, 10.5% at 0.02 (k=4096 projections
75%/38.5%/7.5%; k=12288 gate/up 59.2%/34.1%/17.5%). The tau=0.1 number says
up to ~3x decode from weight-byte skipping is in play if accuracy holds —
the only lever that beats the bandwidth wall. Next: SpQt-style zigzag layout
+ collector + skip kernels, gated on oracle closeness + PPL (not
byte-identity), calibrated against an eval harness.

### MEASURED — LEVERS SPENT OR CLOSED

**Block-size sweep (T2-6)** — the IQ grouped/LDS kernels are block-size
agnostic; 64/128/256/512/1024 swept with byte-identity at every size:
**256 wins or ties everywhere** (512/1024 up to 1.6x worse). grim's
BLOCK_SIZE=256 default stays; this lever is spent.

**LDS-built LUT for IQ3XXS (T3-8) — closed analytically.** trapetum's
additive-VQ LUT needs value-level reuse: dot(a, s⊙grid[c]) with PER-ELEMENT
signs cannot be reused across columns (the sign vector is per-column
weight data), so the LUT build is not amortized at M=1. Post-T1-1 the gap
it targeted is also narrowed (46.5→91 GB/s).

### BLOCKED (tools exist upstream, not on this box)

- **Xtrace (T2-5)**: the paper's tool (g-watch) is not installed;
  /usr/sbin/xtrace is the X11 protocol tracer, unrelated (and X11 is
  deprecated for Wayland anyway). NOT NEEDED today: rocprofv3 kernel-trace
  already delivers per-launch us/VGPR/scratch, the rho microbench gives the
  measured roofline, and the intra-kernel stall gap only matters when
  hand-tuning one kernel body.
- **Kerncap (T2-7)**: AMDResearch/intellikit not installed as a tool — but
  the capability exists locally: `old/repo/Hyperloom-main/` vendors AMD's
  Hyperloom incl. **kernelforge** (agentic kernel-optimization loop:
  kernel_backends incl. HIP, roofline_ceiling, gemm_tune, campaign
  evaluation; intellikit documented as its transitive dep). Caveats: Python
  + agent-SDK driven (claude/codex CLI subprocess), validated on
  MI300X/MI355X (CDNA), targets vLLM/SGLang workloads. grim's cheaper
  equivalent: the kernels are JIT'd from Rust strings, so a grim-side
  `--export-kernel-probe <name>` (kernel name + args + source -> standalone
  hiprtc harness) is an afternoon and needs no Python toolchain.
- **Cache-topology microbench (T3-9 second half)**: no counter access on
  this stack (rocprofv3 --pmc multi-pass impractical, rocprof v1
  unsupported); MT4G's methodology requires porting its suite to RDNA4.

### Standing end-to-end status

Decode-relevant shipped this round: IQ3XXS 46.5→91.3 GB/s and IQ2_S
37.4→81.5 GB/s decode legs (byte-identical). The 9B's Q4_K_M mix carries no
IQ3XXS/IQ2S tensors, so the end-to-end p50 (356.0 ms, 72-token protocol) is
unchanged by design; the wins land on IQ-format checkpoints (Xing4.0-class,
Swift-1.5 27B-class).
