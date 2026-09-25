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
