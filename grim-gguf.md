# `.grim` Format — ROCm-Accelerated GGUF Extension

## Status

Design document. Not yet implemented.

## Guiding Principle

The `.grim` format is a **thin ROCm-native extension layer on top of GGUF**. It is not a competing format — it is a GGUF file with additional metadata that, when loaded on a ROCm-capable grim runtime, enables hardware-aware optimizations unavailable to a vanilla GGUF loader. When loaded on CUDA, CPU, or Metal, grim ignores the ROCm-specific hints and falls back to standard GGUF behavior. No intelligence is lost; performance is *conditionally* gained.

---

## What makes ROCm inference fast or slow?

From research across GGUF, quantization, and ROCm-specific tooling, the bottlenecks are:

1. **Memory bandwidth** — dequantizing weights on-the-fly (the GGUF weight format packs quantized values that must be expanded to f16/f32 before GEMM)
2. **LDS (Local Data Share) pressure** — attention weights must be reorganized into wavefront-sized tiles; naive layouts cause bank conflicts
3. **Instruction-level parallelism** — K-quant block sizes (32 for Q4_K/Q6_K) don't align with ROCm wavefronts (64 threads on RDNA3, 32 on CDNA2)
4. **Mixed-precision routing** — attention projection layers are more quantization-sensitive than FFN layers, but uniform quantization treats them identically
5. **KV cache memory pressure** — context-length scaling kills VRAM; grim-kvquant's Lloyd-Max compression is promising but not integrated at the file-format level

---

## Source Research Summary

### GGUF spec (`ggml-org/ggml/docs/gguf.md`)

GGUF's metadata system is the right place to hang ROCm-specific hints. The spec already defines `general.quantization_version`, `general.architecture`, and per-architecture metadata. We add `grim.rocml.profile` (e.g. `cdna2`, `rdna3`, `mi300x`) and `grim.rocml.wavefront_size: u32` as GGUF metadata keys. This keeps the extension fully compliant — any GGUF reader sees only keys it understands; the rest are silently ignored.

**Design decision**: Store ROCm hints as GGUF metadata fields with `grim.` prefix, not in a separate sidecar file. This preserves single-file deployment.

### K-Quant and I-Quant (`tonisagrista.com`, `localaimaster.com`)

K-quants (Q4_K_M, Q5_K_M, Q6_K) are the right tradeoff for most ROCm hardware. I-quants (IQ4_NL, IQ4_XS) decode slower on CPU but the tradeoff is different on GPU where memory bandwidth is the dominant constraint.

For `.grim`, the recommendation is:
- Default: Q4_K_M for attention tensors, Q5_K_M for embedding/lm_head, Q6_K for output projections
- Optionally include IQ4_XS variants for memory-constrained cards (MI100, MI210) at quality cost
- Support TQ1_0 / TQ2_0 (ternary) for extreme VRAM compression on MI300X where 3-way dot products map well to matrix units

### Advanced GGUF Quantizer (`michaelw9999/advanced-gguf-quantizer`)

Three ideas map directly to `.grim`:

1. **Refined Scale Fit (RSF)**: K-quant scales are re-fitted using importance-matrix data specific to the model's activation distribution, not generic calibration. Grim's `grim-quant` already has `randomized_svd_importance` — this can be the calibration pass.

2. **Activation-aware tensor selection**: For each tensor, multiple candidate encodings (Q4_K, Q5_K, Q6_K, MXFP4) are evaluated and the one with lowest measured activation-error is selected. The result is stored as a per-tensor encoding hint in the `.grim` file.

3. **Mixed NVFP4/MXFP6 tensor types**: Blackwell NVFP4 is not directly portable to ROCm, but the *technique* — per-tensor selection of lower-precision formats — maps well. For ROCm, the equivalent is per-tensor selection between Q4_K, Q5_K, Q6_K, and MXFP4 (GGML type 39).

### GPTQ-GGUF Toolkit (`IST-DASLab/gptq-gguf-toolkit`)

The core insight is **EvoPress**: evolutionary search finds per-layer bitwidth configurations that maximize quality under size constraints.

The workflow:
1. Build a layer database from uniformly quantized GGUFs at multiple bitwidths
2. Run evolutionary search with KL-divergence fitness on calibration data
3. Output a per-tensor bitwidth assignment
4. Stitch the selected layers back into a GGUF

For `.grim`, this workflow becomes a **first-class subcommand** in the grim CLI:

```
grim oxidizer convert Llama-3.1-8B-Q4_K_M.gguf \
    --target grim \
    --calibration-data fineweb_edu_sample.txt \
    --target-bpw 4.0 \
    --generations 50
```

**GPTQ + K-quants outperform K-quants alone** — the error-correcting updates during calibration meaningfully reduce perplexity. Grim's existing GPTQ kernels in `grim-quant` serve as the backend for this calibration pass.

**Design decision**: The `grim oxidizer` tool implements a Rust-native EvoPress equivalent using grim's own tensor engine and calibration datasets. No Python dependency.

### InferMesh (`redbco/infermesh`)

Two ideas map to `.grim`:

1. **Per-GPU tensor partition hints**: For multi-GPU models, store `grim.partition_hint` on tensors indicating which GPU should own them.

2. **Signal-plane metrics baked into the file**: `grim.rocml.profile` includes estimated throughput (tokens/sec per GPU generation), so the runtime scheduler can make informed batching decisions before the first inference.

**Design decision**: Add `grim.gpu_profile` metadata with `gcn_arch`, `estimated_tflops`, `vram_gb` — the runtime uses these for admission control and batch sizing without probing hardware.

### Oxidizer (`microsoft/oxidizer`)

The relevant crates for implementing the oxidizer tool:
- `cachet` for multi-tier caching of compiled kernels
- `thread_aware` for parallel calibration runs
- `seatbelt` for graceful degradation if calibration fails

The oxidizer tool should have a **warm cache** for repeated conversions of the same model architecture.

### gguf-runner (`apimeister/gguf-runner`)

The observation that `RUSTFLAGS="-C target-cpu=native"` gave +20.8% throughput is directly applicable to the ROCm JIT compilation path. Grim's `HsacoKernelCache` and `jit_compile_hsaco` in grim-backend-rocm already do HIP RTC compilation — the JIT path should accept a `target-cpu` hint from the oxidizer to produce wavefront-sized IL0 that matches the target GPU generation.

**Design decision**: The `.grim` file stores `grim.rocml.target_gcn` (e.g. `gfx90a`, `gfx942`, `gfx1100`) so the oxidizer knows which AMD GPU generation to JIT for. The grim runtime uses this for `hiprtc` compilation rather than runtime detection.

---

## Proposed `.grim` Format Specification

### Backward Compatibility

A `.grim` file is a standard GGUF v3 file. The only difference is additional metadata keys with `grim.` prefix. Any GGUF reader skips unknown keys.

### Magic Number

Same as GGUF: `0x465547` ("GGUF"). The file is indistinguishable from a GGUF to a non-grim reader. Grim detects ROCm-specific metadata by checking for the `grim.magic` metadata key — if absent, treat as plain GGUF.

### Additional GGUF Metadata Keys

#### `grim.magic: string`
Value: `"grim-v1"`. Identifies this as a grim-optimized file.

#### `grim.quant_version: uint32`
Oxidizer toolchain version, incremented on format changes.

#### `grim.rocml.profile: string`
Target ROCm profile, e.g. `"cdna2"`, `"rdna3"`, `"mi300x"`, `"all"`.

#### `grim.rocml.wavefront_size: uint32`
Wavefront size of target architecture (32 for CDNA2, 64 for RDNA3).

#### `grim.rocml.target_gcn: string`
AMD GCN architecture identifier, e.g. `"gfx90a"` (MI210/MI250), `"gfx942"` (MI300X), `"gfx1100"` (RX 7900 XTX).

#### `grim.rocml.block_size: uint32`
Recommended thread block size for GEMM kernels.

#### `grim.rocml.lds_size: uint32`
LDS (Local Data Share) size in bytes for target GPU.

#### `grim.rocml.tensor_core_enabled: bool`
Whether to use matrix instruction units (ROCm WMMA/Tensile).

#### `grim.quant_method: string`
e.g. `"evopress-gptq"`, `"uniform-kquant"`, `"importance-matrix"`.

#### `grim.calibration_dataset: string`
Name/path of calibration dataset used for non-uniform quantization.

#### `grim.quant_overrides: array[grim_quant_override_t]`
Per-tensor encoding overrides. Each entry:

```
struct grim_quant_override_t {
    gguf_string_t tensor_name;   // matches tensor name in GGUF tensor_infos
    uint32_t effective_bpw;      // effective bits per weight for this tensor
    uint32_t override_dtype;     // GGML dtype to use (e.g. Q5_K, Q6_K)
    float importance_score;      // from importance matrix calibration
}
```

### ROCm-Specific Tensor Metadata

For each tensor in `gguf_tensor_infos[]`, if `grim.quant_overrides` contains a matching `tensor_name`, use `override_dtype` instead of the GGUF dtype field. This is the mechanism for non-uniform per-tensor quantization.

### Pre-Optimized Layout Hints (future extension)

For tensors where ROCm LDS tiling matters (attention q/k/v/o projections), store an optional `grim.layout_hint`:

- `"wavefront-tiled"`: Reorder weights into wavefront-aligned tiles for LDS efficiency
- `"block-sparse"`: Enable block sparsity pattern for FFN layers

These are execution hints — the grim runtime interprets them but they don't change the tensor data bytes.

---

## `grim oxidizer` Tool Architecture

### Input
Standard GGUF file (e.g. from llama.cpp or Unsloth)

### Pass 1 — Uniform Quantization Baseline
Re-quantize the model to all candidate K-quants (Q2_K through Q6_K) using grim's built-in quantization kernels. Store intermediate results in a temporary layer database.

### Pass 2 — Importance Matrix Generation
Run a calibration dataset (e.g. FineWeb-Edu, 2M tokens) through the f16 baseline model. Collect per-layer activation statistics to compute an importance ranking. This replaces the naive "all layers equal bitwidth" approach.

### Pass 3 — EvoPress Search
Run evolutionary search over per-tensor bitwidth assignments:
- Population: 128 configurations
- Fitness: KL-divergence on perplexity against baseline
- Budget: constrain average BPW to target (e.g. 4.0)
- Selection: 3-stage with increasing calibration tokens (2K → 16K → 128K)

Output: per-tensor encoding map.

### Pass 4 — Per-Tensor Re-Quantization
For each tensor, re-quantize to the selected encoding using the GPTQ error-correcting update pass (grim-quant already has the kernels).

### Pass 5 — ROCm Profile Generation
For the target GCN architecture, determine wavefront size, LDS size, block size, and matrix unit availability. Encode as metadata.

### Pass 6 — Write `.grim`
Assemble the optimized tensor data and grim metadata into a GGUF-compatible file.

---

## Key Differences from Standard GGUF

| Property | Standard GGUF | `.grim` (ROCm-optimized) |
|---|---|---|
| Per-tensor encoding | Uniform (one dtype for all) | Non-uniform (EvoPress-selected per tensor) |
| Calibration | None (or generic imatrix) | Importance-matrix with task-specific data |
| ROCm hints | None | Wavefront size, GCN arch, block size, LDS size |
| Scale fitting | K-quant default | Refined Scale Fit (RSF) per tensor |
| Attention tensor precision | Same as FFN | Higher precision for attention projections |
| MXFP4 support | Via GGML type 39 | Via GGML type 39 + ROCm GEMM path |
| Memory layout | Generic | Wavefront-tiled for attention layers |
| Calibration metadata | None | Stored for reproducibility |

---

## Compatibility Matrix

| Runtime | `.grim` behavior |
|---|---|
| grim on ROCm | Full optimization — non-uniform quantization, ROCm hints, wavefront tiling |
| grim on CUDA | Ignore `grim.*` metadata — treat as standard GGUF; falls back to CUDA kernels |
| grim on CPU | Ignore `grim.*` metadata — standard GGUF dequantization |
| llama.cpp / Ollama | Ignores `grim.*` keys — loads as vanilla GGUF |
| Other GGUF loaders | Standard GGUF behavior |

---

## What NOT to do

- **Don't invent a new magic number** — the whole point is being loadable as GGUF
- **Don't pre-compile GPU kernels into the file** — keep the `.grim` as model data only; JIT compilation is the right path (grim already has HsacoKernelCache)
- **Don't require ROCm for loading** — fail-closed for missing GPU is acceptable, but the file must be parseable without ROCm
- **Don't hardcode block sizes** — ROCm architectures differ; store the profile metadata and let the runtime decide
- **Don't do calibration at runtime** — the oxidizer tool does this once; the runtime just reads the pre-computed overrides

---

## Phased Implementation Plan

### Phase 1 — Metadata Extension (foundation)

- Extend `GgufProvider` / `GgufFile` in `grim-format` to read `grim.*` metadata keys
- Parse `grim.quant_overrides` into a lookup table keyed by tensor name
- Add `grim_quant_override_t` struct matching the GGUF array encoding
- Add override resolution to tensor loading path in `grim-backend-rocm`
- Update `map_gguf_dtype_to_grim()` to check for overrides
- **Verify**: plain GGUF still loads identically; `.grim` with overrides returns correct dtype per tensor

### Phase 2 — Importance-Matrix Calibration

- Implement `grim-quant` GPTQ path with error-correcting updates for K-quants
- Add `randomized_svd_importance` calibration pass using grim tensor engine
- Create `grim oxidizer calibrate` subcommand with dataset loading
- Output: importance scores per tensor, stored in `grim.quant_overrides[].importance_score`
- **Verify**: per-tensor importance scores correlate with known sensitive layers (attn_q, attn_k)

### Phase 3 — EvoPress Search Engine

- Implement evolutionary search as `grim oxidizer search` subcommand
- Population: 128 offsprig per generation, 50 generations default
- 3-stage selection: 2K → 16K → 128K calibration tokens
- Fitness: KL-divergence on perplexity vs baseline
- Output: per-tensor encoding map (matching `grim.quant_overrides`)
- **Verify**: search output on Llama 3.1 8B matches published EvoPress results at ~4.0 BPW

### Phase 4 — ROCm Profile Hints

- Extend `RocmDevice` to read `grim.rocml.profile` from file metadata
- Add `target_gcn`, `wavefront_size`, `lds_size`, `block_size` fields to `RocmDevice`
- Use these in GEMM kernel selection (lookup_gemm_config, rocblas_gemm_ex path)
- **Verify**: same `.grim` on MI300X (gfx942) vs MI210 (gfx90a) picks correct kernel config

### Phase 5 — Wavefront-Tiled Layout

- Implement weight reorganization pass in oxidizer for attention projections
- Add `grim.layout_hint` parsing and storage in `grim.quant_override_t`
- Implement matching dequantization kernel in `grim-backend-rocm`
- **Verify**: attention projection dequantization is LDS-bank-conflict-free on RDNA3

### Phase 6 — End-to-End CLI

- Wire all passes into `grim oxidizer convert` single command
- Add warm cache for repeated conversions (same model architecture)
- Add `--target grim` / `--output model.grim` flags
- Add `grim info` subcommand to dump grim metadata without loading model
- **Verify**: `grim oxidizer convert Llama-3.1-8B-Q4_K_M.gguf --target grim` produces a `.grim` that loads on ROCm and reports correct profile

---

## Relevant Workspace Files

| File | Role |
|---|---|
| `crates/grim-format/src/gguf.rs` | GGUF parsing — add `grim.*` metadata keys here |
| `crates/grim-format/src/tprov.rs` | `GgufProvider` — tensor loading — add override resolution here |
| `crates/grim-tensor/src/dtype.rs` | `DType`, `KQuantScheme`, `QuantProvenance` — already defined |
| `crates/grim-quant/src/lib.rs` | Q4_K/Q8_0/GPTQ kernels — already exists; add RSF fitting here |
| `crates/grim-backend-rocm/src/lib.rs` | `RocmDevice`, `RocmStorage`, `rocblas_gemm_ex` — add profile hints here |
| `crates/grim-backend-rocm/src/hip.rs` | HIP FFI — likely no changes needed |
| `crates/grim-kvquant/src/lib.rs` | KV cache compression — separate from file format but complementary |
| `crates/grim-cli/src/main.rs` | Add `oxidizer` subcommand group here |
| `crates/grim-cli/src/oxidizer.rs` | New file — EvoPress search + calibration orchestration |

---

## References

- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp K-quants: https://github.com/ggerganov/llama.cpp
- Advanced GGUF quantizer (NVFP4/MXFP6): https://github.com/michaelw9999/advanced-gguf-quantizer
- GPTQ-GGUF toolkit (EvoPress): https://github.com/IST-DASLab/gptq-gguf-toolkit
- InferMesh (GPU-aware routing): https://github.com/redbco/infermesh
- Oxidizer crates (Microsoft): https://github.com/microsoft/oxidizer
- gguf-runner: https://github.com/apimeister/gguf-runner