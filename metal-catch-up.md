# Metal backend catch-up plan

From: deficiency audit (ROCm vs CUDA/Vulkan/CPU/Metal). Grounded in `crates/grim-backend-metal/src/lib.rs` (MetalDevice, MetalDeviceInner, MetalPipelines, Tuner::search_tile_config, MetalStorage, dequantize_*_host) and the audit's transfer IDs T1–T7.

## 1. Current state (what Metal actually has)

- Device: `MetalDevice` wraps ordinal + `MetalDeviceInner` (MTLDevice, MTLCommandQueue, MetalPipelines, active command buffer). `MetalDevice::probe` returns device list via `MetalContext`. Apple-only (`#[cfg(target_vendor = "apple")]`); non-Apple falls back to a CPU-memory stub.
- Pipeline set (MetalPipelines) — **largest fixed pipeline set of any backend**: matmul + qkv_attn / qkv_paged_attn / tree_attn / kv_dequant_attn / mul_scalar + quant_mxfp8 / quant_q4k + dequant_mxfp8 / dequant_q8_0 / dequant_iq3s / dequant_iq4nl / dequant_iq4xs + moe_fused_dispatch + **add_rms_norm** (fused op ROCm doesn't have yet). Pipelines are precompiled MSL from `include_str!("kernels.msl")` / `kernels.metallib`; the pipeline for each op is looked up by name (`get_pipeline("grim_...")`). There is **no runtime MSL→metallib compile per candidate** today (metal-rs compile exists as a build-time path, not a per-candidate runtime path in the crate).
- Autotuner: `Tuner::search_tile_config` — heuristic only. Picks Matmul64 / Matmul32 / Matmul64Bf16 by block_m; **no GPU timing, no compile-per-candidate**.
- Matmul path: `BackendDevice::matmul` (Apple) uses `cblas_sgemm` via Accelerate as a device-absent fallback and otherwise dispatches through the MSL matmul pipeline; non-Apple is CPU-memory stub.
- Quant/dequant: MetalPipelines quant/dequant pipelines above + broad **host** dequant (`dequantize_q8_0_host`, `q4k_host`, `iq2xxs/iq2xs/iq2s/iq3xxs/iq3s/iq4nl_host`, `fp8_host`, `mxfp4_host`, `mxfp8_host`). quantize_on_device, moe_fused_dispatch.
- Attention: `MetalDevice::qkv_attention` (MSL grim_qkv_attention), qkv_paged_attn / tree_attn / kv_dequant_attn pipelines.
- Fused: `MetalDevice::fused_add_rms_norm` (grim_add_rms_norm) — this op exists on Metal and **not yet on ROCm**.
- No: capability profiler, HW fingerprint in cache key, empirical autotuner, quant-capability gate (family-gated), multi-GPU/collective/P2P (Metal has no cross-GPU peer primitive), split-K, TLOLog/op-identity classifier, shared-memory resource gate, graph capture.

## 2. Gaps (relative to ROCm)

- No JIT/MSL-per-candidate compile (fixed .metallib pipelines only).
- Autotuner is heuristic (Matmul64/32/64Bf16 by block_m) — does not **measure** on the real GPU.
- No `ShapeClass`/op-identity classifier → lm_head lands by m.
- No MetalCaps struct folding MTLDevice properties (gpuFamily, registryID, maxBufferRowAlignMask, maxThreadgroupMemoryLength, maxTotalThreadgroupMemory, family→precision/FP8 support) into a caps struct with epoch + supports(mode).
- No HW fingerprint in cache key (keyed by pipeline name / metallib hash only).
- No split-K, no TLOLog tile, no threadgroup-memory resource gate.
- No multi-GPU/collective/P2P (architecturally blocked — Metal has no cross-GPU peer primitive; even within one Mac, multiple GPUs are not a peer-transferrable setup in Metal the way ROCm/xGMI/PCIe is).
- Apple-only constraint is inherent to Metal, not a deficiency.

## 3. Transfers to apply (from ROCm)

### T6-immediate-subset — empirical autotuner over existing pipelines (no precondition) — HIGHEST Metal value
Metal already has a small fixed candidate set (Matmul64 / Matmul32 / Matmul64Bf16) and a real device. Convert `Tuner::search_tile_config` from heuristic to **measured**: time the three existing MSL matmul pipelines on the real GPU for the real (m,n,k), pick the fastest, persist in a tune cache keyed by `(device_registryID+caps_hash, m, n, k)`. No re-compile needed — precondition-free. This is the precondition-free slice of ROCm's FCP and validates "measure, don't estimate" on Metal's real pipelines first.

Sub-point: the BF16 vs F32 pipeline selection (Matmul64Bf16 vs Matmul64/Matmul32) is currently by dtype; fold the measured decision in so both BF16 and F32 variants are timed when relevant.

### T3 — op-identity ShapeClass + TLOLog (no precondition)
`Tuner::search_tile_config(m,n,k)` bins by m only. Add a shape/op tag forwarded from the matmul call site:
- `ShapeClass` enum (Decode/Prefill/TLOLog) + `GemmOp` (Attention/Ffn/LmHead/Other) + `from_op(op,m)` — shared shapes (gemm_tuning/autotune), reused by Metal.
- TLOLog arm → wide-N tile (block_m=16, block_n=64, block_k=64). Metal's existing matmul pipelines are Matmul64/Matmul32/Matmul64Bf16; for TLOLog wide-N the natural mapping on the existing surface is to use the **Matmul64** pipeline with the (16,64) tile, or add a `MatmulLmHead` MSL variant later. For now: tag it TLOLog and let the measured autotuner (T6-immediate) pick the best of the existing pipelines for that wide-N shape.

### T1 — MetalCaps struct + epoch + supports(mode)
Fold MTLDevice properties — `gpuFamily` (family2 => MTLFamily2Device, precision/format support), `registryID`, `maxBufferRowAlignMask`, `maxThreadgroupMemoryLength`, `maxTotalThreadgroupMemory`, `supportsFamily2`, and any fp8/bf16 precision support the family provides — into a `MetalCaps` struct with an epoch counter; add `supports(mode: QuantMode) -> bool` (mxfp8/fp8/bf16 availability by family) and gate pipeline selection on it. Pattern: ROCm's `QuantCapability::supports(mode)`. Concretely: mxfp8/fp8 pipelines should only be selected when the device family supports them; otherwise fall back to F32/BF16.

### T2 — HW fingerprint in cache key
Metal cache is keyed by pipeline name / metallib hash only. For pipeline-selection variants (Matmul64 vs 32 vs 64Bf16) that depend on device family/caps, add `device_registryID` + `caps_hash` to the key so the same op on a different Apple GPU (e.g. M1 vs M3 family) doesn't silently pick the wrong variant. Value is lower than ROCm's compile-key fingerprint (pipelines are fixed) but matters for **selection** that depends on family/caps.

### T4 — threadgroup-memory resource gate (for custom kernels / future tile search)
Metal threadgroup memory is bounded by `MTLDevice::maxThreadgroupMemoryLength` (read from the device). Gate Metal tile/workgroup configs by that per-device threadgroup-memory ceiling and by `maxTotalThreadgroupMemory`, reject overcommit before dispatch. Analog of ROCm's #6 `candidate_valid`. Precondition-free; useful whenever Metal adds tile variants beyond the fixed pipelines.

### T7 — MSL specialization as the Metal-native "per-candidate variant" (precondition: shader designed for specialization)
ROCm's `compute_kernel_source_with_spec` (per-candidate #define injection) has a Metal-native analog: **MSL function constants / specialization constants** (compile-time-specialized per pipeline creation, not per full metallib re-compile) and MSL argument buffers for per-launch parameters. If the existing matmul MSL shader is refactored to expose block_m/block_n/block_k/split_k as function/specialization constants rather than hardcoded, then the Metal autotuner can create a specialized pipeline per candidate **without re-compiling the whole metallib** — the precondition for T6-full + T7 becomes "MSL shader designed for specialization/function constants," which is a shader-design change, lighter than a full metal-rs per-candidate compile. This is the right Metal-native route to per-candidate variants.

### T6-full — blocked until MSL specialization/function constants exist
The full FCP (block_k∈{16,32,64,128}, split_k, time each, persist) needs per-candidate variants. On Metal the cheapest path is MSL function/specialization constants (T7). Until the matmul MSL exposes those constants, the full candidate generation is blocked; the immediate-subset (T6-immediate) over the 3 existing pipelines does not need it.

### Not transferrable from ROCm to Metal
- hiprtc/rocBLAS/RCCL/P2P/GCN-gating/hip-graph — ROCm-specific. Metal's analogs: metal-rs compile for MSL (build-time, not per-candidate runtime today), no BLAS library (matmul is a custom MSL kernel, no rocBLAS equivalent), no cross-GPU collective primitive (architecturally blocked — Metal has no peer transfer across GPUs; multi-GPU is not a Metal path here), no GCN arch gating (Metal gates on device family/registryID). These are either not applicable or blocked; skip.
- mxfp8/mxfp4/fp8 quant pipelines: Metal already ships these as fixed MSL pipelines — no need to transfer ROCm's charon quant kernels; Metal's coverage is comparable.

### Transfers TO ROCm (Metal → ROCm, for completeness)
- **fused add_rms_norm** — Metal has `grim_add_rms_norm` MSL pipeline; ROCm does not yet (no charon kernel). Highest-value **one-directional** transfer: port the fused op to a charon HIP kernel on ROCm. This is Metal→ROCm, not ROCm→Metal.
- Broader fixed-blob quant/dequant family: Metal ships Q8_0/FP8/MXFP4/MXFP8/Q4K + dequant iq3s/iq4nl/iq4xs as precompiled MSL; ROCm's quant surface is charon HIP kernels with comparable breadth. Not a real gap.
- Attention breadth: Metal has qkv_paged_attn + tree_attn + kv_dequant_attn pipelines — ROCm's attention is the planned charon qkv_attention kernel (paged/tree not yet in the plan). If paged/tree attention is wanted on ROCm, port from Metal's MSL shaders (rewrite to HIP, not a literal transfer).

## 4. Ordering

1. T6-immediate-subset (empirical time the 3 existing matmul pipelines + persist) — highest Metal value, no precondition.
2. T3 — op-identity + TLOLog tag forwarded to Metal Tuner (so lm_head wide-N shapes get the right candidate set).
3. T1 — MetalCaps + epoch + supports(mode) (gates mxfp8/fp8/bf16 pipeline availability by family).
4. T2 — HW fingerprint in cache key (selection variants that depend on family/caps).
5. T4 — threadgroup-memory resource gate (overcommit guard for tile variants).
6. T7 — refactor matmul MSL to expose block_m/block_n/block_k/split_k as function/specialization constants (precondition for per-candidate variants).
7. T6-full — per-candidate variants via MSL specialization/function constants + time + persist; deferred until T7 done.
8. Split-K MSL variant — deferred (needs a K-split MSL path; lower priority).
9. **Metal→ROCm**: port fused add_rms_norm to a charon HIP kernel on ROCm (separate, one-directional).

## 5. Validation

- `Tuner::search_tile_config(m,n,k)` returns a tile and picks the matmul pipeline by **measured GPU time**, not by block_m heuristic; repeat shape hits the cache (no re-time).
- BF16 vs F32 pipeline choice reflects measured time when both are applicable, not dtype alone.
- `MetalCaps::supports(mxfp8)` is true only on device families that support it; family without it does not select the mxfp8 pipeline.
- Cache key includes device_registryID + caps_hash; different Apple GPU → distinct key.
- Tile variants (if any) are rejected by `maxThreadgroupMemoryLength` + `maxTotalThreadgroupMemory` before dispatch.
- On non-Apple, Metal falls back to CPU-memory stub — the catch-up above applies to the Apple Metal path; non-Apple is not a real GPU backend.

## 6. Scope note

Metal is Apple-only and single-device in this crate — there is **no cross-GPU collective/peer primitive** in Metal, so multi-GPU kernel launch, RCCL-style collectives, and P2P routing are **out of scope** for Metal (they don't exist in the platform here). The Metal catch-up is about making the single-device pipeline selection empirical, family/caps-gated, HW-fingerprinted, and resource-valid, plus the one-metal-has-it-ROCm-doesn't transfer (fused add_rms_norm) back to ROCm.
