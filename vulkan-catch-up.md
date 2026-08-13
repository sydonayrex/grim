# Vulkan backend catch-up plan

From: deficiency audit (ROCm vs CUDA/Vulkan/CPU/Metal). Grounded in `crates/grim-backend-vulkan/src/lib.rs` (VulkanDevice, VulkanStorage, VulkanAutotuner, VulkanKernel enum, spirv_for) and the audit's transfer IDs T1–T7.

## 1. Current state (what Vulkan actually has)

- Device: `VulkanDevice` wraps a single physical device + device + queue; `VulkanDevice::probe` returns physical devices with queue-family/memory-heap properties. `VulkanStorage` wraps VkBuffer + device + physical_device + memory.
- Kernel execution: **fixed precompiled SPIR-V blobs** selected by the `VulkanKernel` enum (`spirv_for(kernel)`). There is **no runtime source→SPIR-V compile** path in the crate (no glslc/SPIRV-Gen per candidate today).
- Autotuner: `VulkanAutotuner::search_tile_config(m, n, k)` — heuristic only. Picks `Matmul64` / `Matmul32` / `Matmul64Bf16` by block_m heuristics; **no GPU timing, no compile-per-candidate**.
- Matmul path: `BackendDevice::matmul` selects the SPIR-V blob by dtype (BF16 path → Matmul64Bf16; else by tile_config.block_m 64 vs 32), runs `run_compute_shader` with push constants + grid_x/grid_y from the tile config.
- Quant/dequant on-device: VulkanKernel quant/dequant SPIR-V blobs (Q8_0, FP8, MXFP4, MXFP8, Q4K, Q5K, Q6K, Q3K(+placeholder)), quantize_on_device, dequantize paths, moe_fused_dispatch (SPIR-V), and dequant mirrors.
- Attention: `VulkanDevice::qkv_attention_inner` (QkvAttention / QkvAttentionSwa SPIR-V), BackendDevice dequantize_qkv_attention; kv_dequant_attn pipeline.
- Scalar helpers: mul_scalar pipeline (grim_mul_scalar).
- No: capability profiler, HW fingerprint in cache key, empirical autotuner, quant-capability gate, multi-GPU/collective/P2P (Vulkan has no cross-GPU collective primitive), split-K, TLOLog/op-identity classifier, shared-memory resource gate, graph capture.

## 2. Gaps (relative to ROCm)

- No JIT/source compile (fixed SPIR-V blobs only).
- Autotuner is heuristic (Matmul64/32/64Bf16 by block_m) — does not **measure** on the real GPU.
- No `ShapeClass`/op-identity classifier → lm_head lands by m.
- No VulkanCaps struct folding physical-device properties (extension/feature support, shared memory size, max workgroup size) into a caps struct with epoch + supports(mode).
- No HW fingerprint in cache key (cache keyed by VulkanKernel enum only).
- No split-K, no TLOLog tile, no shared-memory resource gate.
- No multi-GPU/collective/P2P (architecturally blocked — Vulkan has no cross-GPU peer primitive).

## 3. Transfers to apply (from ROCm)

### T6-immediate-subset — empirical autotuner over existing blobs (no precondition) — HIGHEST Vulkan value
Vulkan already has a small fixed candidate set (Matmul64 / Matmul32 / Matmul64Bf16) and a real device. Convert `VulkanAutotuner::search_tile_config` from heuristic to **measured**: time the three existing SPIR-V blobs on the real GPU for the real (m,n,k), pick the fastest, persist in a tune cache keyed by `(physical_device_handle+caps_hash, m, n, k)`. No re-compile needed — precondition-free. This is the precondition-free slice of ROCm's FCP and validates "measure, don't estimate" on Vulkan's real pipelines first.

Sub-point: the BF16 vs F32 blob selection is currently by dtype+surface size; fold that into the measured decision (time both BF16 and F32 variants when relevant) rather than always picking Matmul64Bf16 when any input is BF16.

### T3 — op-identity ShapeClass + TLOLog (no precondition)
`VulkanAutotuner::search_tile_config(m,n,k)` bins by m only. Add a shape/op tag forwarded from the matmul call site:
- `ShapeClass` enum (Decode/Prefill/TLOLog) + `GemmOp` (Attention/Ffn/LmHead/Other) + `from_op(op,m)` — these live in the shared shapes (gemm_tuning/autotune), reused by Vulkan.
- TLOLog arm → route to a wide-N tile. Vulkan's existing blobs are Matmul64/Matmul32/Matmul64Bf16; for TLOLog (N=vocab dominant, block_m=16, block_n=64) the natural mapping on the existing surface is to use the **Matmul64** blob with the (16,64) tile, or add a `MatmulLmHead` variant later. For now: tag it TLOLog and let the measured autotuner (T6-immediate) pick the best of the existing blobs for that wide-N shape — the empirical timing is the real arbiter.

### T1 — VulkanCaps struct + epoch + supports(mode)
Fold `VkPhysicalDeviceProperties` + `VkPhysicalDeviceMemoryProperties` + **extension/feature queries** (VK_KHR_fp8 / shaderFloatControls / SPI, shared memory size, maxComputeWorkGroupSize, maxMemoryAllocationSize, FP/BF16 shader support) into a `VulkanCaps` struct with an epoch counter; add `supports(mode: QuantMode) -> bool` (fp8 shaders present, BF16 support, etc.) and gate blob/pipeline selection on it. Pattern: ROCm's `QuantCapability::supports(mode)`.

### T2 — HW fingerprint in cache key
Vulkan cache is keyed by `VulkanKernel` enum only. For pipeline/selection variants (Matmul64 vs 32 vs 64Bf16) that depend on device caps (shared memory size, workgroup limits, FP/BF16 support), add the caps hash + physical device identity to the key so the same op on a different GPU doesn't silently pick the wrong variant. Value is lower than ROCm's compile-key fingerprint (Vulkan blobs are fixed) but still matters for **selection** that depends on caps.

### T4 — shared-memory resource gate (for custom kernels / future tile search)
Vulkan workgroup shared memory is bounded by `VkPhysicalDeviceLimits::maxComputeSharedMemorySize` (typically 16–32 KB on Vulkan GPUs — different from ROCm's 64 KB, but the pattern is identical). Gate Vulkan tile/workgroup configs by that per-device shared-memory ceiling and by `maxComputeWorkGroupSize` (per-workgroup thread ceiling), reject overcommit before dispatch. Analog of ROCm's #6 `candidate_valid`. This is the precondition-free resource-validity gate; useful whenever Vulkan adds tile variants beyond the fixed blobs.

### T7 — specialization constants as the Vulkan-native "per-candidate variant" (precondition: shader designed for specialization)
ROCm's `compute_kernel_source_with_spec` (per-candidate #define injection) has a Vulkan-native analog that doesn't require re-compiling the whole SPIR-V: **specialization constants** (push constant / specialization-constant values baked at pipeline-creation time, not at shader-compile time). If the existing matmul SPIR-V shaders are refactored to expose block_m/block_n/block_k/split_k as specialization constants rather than hardcoded, then the Vulkan autotuner can create a specialized pipeline per candidate **without re-compiling from GLSL** — the precondition for T6-full + T7 becomes "shaders designed for specialization constants," which is a shader-design change, lighter than a full glslc-per-candidate compile. This is the right Vulkan-native route to per-candidate variants; full re-compile per candidate is the wrong route on Vulkan.

### T6-full — blocked until specialization constants or re-compile path exists
The full FCP (generate block_k∈{16,32,64,128}, split_k, time each, persist) needs per-candidate variants. On Vulkan the cheapest path is specialization constants (T7). Until the shaders expose those constants, the full candidate generation is blocked; the immediate-subset (T6-immediate) over the 3 existing blobs does not need it.

### Not transferrable from ROCm to Vulkan
- hiprtc/rocBLAS/RCCL/P2P/GCN-gating/hip-graph — ROCm-specific. Vulkan's analogs: glslc/SPIRV-Gen for compile (not used today), no BLAS library (the matmul is a custom SPIR-V kernel, no rocBLAS equivalent to port), no cross-GPU collective primitive (architecturally blocked — Vulkan has no peer transfer across GPUs), no GCN arch gating (Vulkan gates on device families/extensions). These are either not applicable or blocked; skip.

### Transfers TO ROCm (Vulkan → ROCm, for completeness)
- Broader fixed-blob quant/dequant family (Vulkan ships Q8_0/FP8/MXFP4/MXFP8/Q4K/Q5K/Q6K/Q3K + dequant mirrors as precompiled SPIR-V) — ROCm's quant surface is charon HIP kernels with comparable breadth, different delivery; not a real gap.
- mul_scalar pipeline (functionally equivalent to ROCm's bare-metal charon dispatches).

## 4. Ordering

1. T6-immediate-subset (empirical time the 3 existing matmul blobs + persist) — highest Vulkan value, no precondition.
2. T3 — op-identity + TLOLog tag forwarded to VulkanAutotuner (so lm_head wide-N shapes get the right candidate set).
3. T1 — VulkanCaps + epoch + supports(mode) (gates BF16/FP8 blob availability).
4. T2 — HW fingerprint in cache key (selection variants that depend on caps).
5. T4 — shared-memory resource gate (overcommit guard for tile variants).
6. T7 — refactor matmul SPIR-V to expose block_m/block_n/block_k/split_k as specialization constants (precondition for per-candidate variants).
7. T6-full — per-candidate variants via specialization constants + time + persist; deferred until T7 done.
8. Split-K shader variant — deferred (needs a K-split SPIR-V path; lower priority than wide-N/TLOLog).

## 5. Validation

- `VulkanAutotuner::search_tile_config(m,n,k)` returns a tile and picks the matmul blob by **measured GPU time**, not by block_m heuristic; repeat shape hits the cache (no re-time).
- BF16 vs F32 blob choice reflects measured time when both are applicable, not dtype alone.
- `VulkanCaps::supports(fp8_shader)` gates FP8 blob selection; device without FP8 shader does not try the FP8 path.
- Cache key includes physical_device_handle + caps_hash; different Vulkan GPU → distinct key.
- Tile variants (if any) are rejected by `maxComputeSharedMemorySize` + `maxComputeWorkGroupSize` before dispatch.

## 6. Scope note

Vulkan is single-device by nature in this crate — there is no cross-GPU collective primitive in Vulkan, so multi-GPU kernel launch, RCCL-style collectives, and P2P routing are **out of scope** for Vulkan (they don't exist in the platform here). The Vulkan catch-up is about making the single-device pipeline selection empirical, capability-gated, HW-fingerprinted, and resource-valid.
