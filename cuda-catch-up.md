# CUDA backend catch-up plan

From: deficiency audit (ROCm vs CUDA/Vulkan/CPU/Metal). Grounded in `crates/grim-backend-cuda/src/lib.rs` (BackendDevice impl, launch_* surfaces, cublas path) and the audit's transfer IDs T1–T7.

## 1. Current state (what CUDA actually has)

- Device: `CudaDevice` wraps ordinal + pooled CUBLAS handle; `CudaDevice::probe` returns ordinals, `CudaHandle` loads PTX/CUBIN via `cuModuleLoad`/`cuModuleGetFunction`/`cuModuleLaunchKernel`. There is **no runtime source→binary compile** path in the crate (no cuJit/IC usage today).
- GEMM: `BackendDevice::matmul` is a single `cublasSgemm_v2` call with a hardcoded row/col-major transposition trick. Tile selection lives in `gemm_tuning.rs::lookup_gemm_config(m, n, k, wave)` — W64/W32 tables, split_k hint for k-heavy decode only; **no HW caps feed tile choice**.
- Quant/dequant on-device: `launch_quant_q8_0`, `launch_quant_fp8`, `launch_fused_quant_gemm`, `launch_dequant_generic/fp8/mxfp4/mxfp8/mxfp_kernel`, `dequantize_on_device`; broad **host** dequant (`dequantize_q8_0_host`, `q4k_host`, `iq2xxs/iq2xs/iq2s/iq3xxs/iq3s/iq4nl_host`).
- Attention: `CudaDevice::qkv_attention` (launch_qkv_*), BackendDevice::dequantize_qkv_attention, qkv_attention_paged.
- MoE: `CudaDevice::moe_fused_dispatch`.
- Small custom kernels: `launch_rank1_kernel` for `grim_add`/`grim_mul`/silu_mul (register 1D kernels from PTX).
- No: capability profiler, HW fingerprint in cache key, autotuner, quant-capability gate, multi-GPU/NCCL wiring, P2P topology, split-K derivation, TLOLog/op-identity classifier, graph capture, shared-memory resource gate.

## 2. Gaps (relative to ROCm)

- No JIT source parametrization (static PTX only).
- No empirical autotuner (cuBLAS chooses internally; no user-side tile search or measurement).
- No `ShapeClass`/op-identity classifier → lm_head always gets Decode/Prefill by m.
- No compute-capability / shared-mem / max-threads folding into a caps struct or cache key.
- No split-K derivation from real K.
- No multi-GPU/NCCL/P2P (architecturally blocked until NCCL is wired — lower priority).
- No capability/epoch gating of quant formats.

## 3. Transfers to apply (from ROCm)

Precedence: precondition-free first.

### T3 — op-identity ShapeClass + TLOLog (no precondition)
**Highest immediate value.** `gemm_tuning.rs::lookup_gemm_config` bins only by m/wave. Add:
- `ShapeClass` enum (Decode/Prefill/TLOLog) and `GemmOp` enum (Attention/Ffn/LmHead/Other) + `ShapeClass::from_op(op, m)` to the CUDA-facing reuse of gemm_tuning (the tables are shared across backends in this repo, so this mainly lands in the shared `gemm_tuning.rs`/`autotune.rs` shapes, not a CUDA-only file).
- A TLOLog arm: block_m=16, block_n=64, block_k=64 — justified (N=vocab dominant, K=hidden reused across wide N).
- Forward a `GemmOp` tag from whichever CUDA matmul/attention call sites produce lm_head so lm_head is TLOLog regardless of m.

Concrete effect: CUDA's lm_head (single cuBLAS call, no tile search) finally gets a tile/algo hint that matches its wide-N profile instead of landing in Decode (16,16) or Prefill (32,32).

### T1 — capability struct + epoch + supports(mode)
Fold `cudaDeviceProp` fields (computeCapabilityMajor/Minor, sharedMemPerBlock, maxThreadsPerBlock, multiProcessorCount, memPitch, totalGlobalMem) + runtime sampled info into a `CudaCaps` struct with an epoch counter; add `supports(mode: QuantMode) -> bool` (fp8-native on compute≥8.0, etc.) and gate launch paths on it. Pattern: ROCm's `QuantCapability::supports(mode)` / `resolve_quant_mode`.

### T2 — HW fingerprint in cache key
CUDA loads PTX per-device and keys by `(entry, gpu_target, source_hash)`. Add device caps to the key: `(ptx_name, ordinal, compute_capability, sharedMemPerBlock, maxThreadsPerBlock, multiProcessorCount)` so a different GPU doesn't silently get the wrong launch config.

### T4 — shared-memory resource gate (for custom kernels)
For the `launch_rank1_kernel` paths and any future custom PTX kernels (not cuBLAS, which manages occupancy internally): gate tile/workgroup configs by the **per-SM shared memory budget** = `sharedMemPerBlock * multiProcessorCount` (the per-block figure is a request ceiling; co-residency is bounded per-SM) and by `maxThreadsPerBlock`. Reject overcommit before launch. Analog of ROCm's #6 `candidate_valid`.

### T6-immediate-subset — empirical autotuner over existing blobs (no precondition)
CUDA's candidate set today is small (cuBLAS does its own selection; custom kernels are few). The precondition-free slice: for any **custom** PTX kernels that have multiple variants (e.g. add/mul/silu rank-1 paths), time them on the real GPU for the real shape and pick the faster — persist in a small tune cache keyed by `(ordinal, caps_hash, op, m/n/k shape)`. This is the precondition-free slice of ROCm's FCP. Value is lower than Vulkan/Metal here because CUDA's main GEMM path is cuBLAS (already tuned), but it still validates the "measure, don't estimate" pattern on custom kernels.

### T6-full + T7 — blocked (precondition: runtime compile/specialization)
ROCm's full FCP (compile+time+persist per candidate) and `compute_kernel_source_with_spec` (per-candidate #define injection) require a runtime compile path CUDA doesn't have in this crate today. Closest analogs:
- PTX #define specialization via CUDA JIT (cuJit/IC, or PTX constant substitution at load).
- But grim's CUDA crate doesn't use a JIT compile path at all — so this transfer is **blocked** until a cuJit/IC or PTX-constant-substitution path is added. Lower priority than Vulkan/Metal here because CUDA's GEMM is cuBLAS (which already tunes internally).

### Not transferrable from ROCm to CUDA
- hiprtc/rocBLAS/RCCL/P2P/GCN-gating/hip-graph — ROCm-specific. CUDA has its own analogs (cuJit, cuBLAS internal selection, NCCL exists, cudaDeviceCanAccessPeer) but grim's CUDA crate doesn't wire them; where they don't exist at all in the crate, they're lower priority because the GEMM path is cuBLAS and multi-GPU isn't a CUDA-crate concern today.

### Transfers TO ROCm (CUDA → ROCm, for completeness)
- None material — CUDA has nothing ROCm lacks here. (CPU/Vulkan/Metal feed ROCm; CUDA doesn't.)

## 4. Ordering

1. T3 (op-identity + TLOLog to shared shapes) — biggest concrete win, no precondition.
2. T1 (CudaCaps + epoch + supports(mode)).
3. T2 (HW fingerprint in cache key).
4. T4 (shared-memory resource gate for custom kernels).
5. T6-immediate-subset (empirical time for custom PTX variants + cache).
6. T6-full/T7 — deferred until a cuJit/IC or PTX-constant-substitution path exists; low priority because GEMM = cuBLAS.

## 5. Validation

- `lookup_gemm_config` with a TLOLog shape/op tag returns block_m=16, block_n=64, block_k=64 (not 16,16 or 32,32).
- `CudaCaps::supports(fp8_native)` is true only when compute≥8.0; false before (gates the fp8 launch paths).
- Cache key includes caps fields; two GPUs with different sharedMemPerBlock/maxThreadsPerBlock get distinct keys.
- Custom rank-1 kernel variants that differ in block size are timed and the faster is cached; idempotent on repeat shape.

## 6. Scope note

This plan is about the CUDA crate's **device/kernel/dispatch** surface, not about wiring NCCL multi-GPU into the CUDA crate (out of scope; multi-GPU collectives in this repo are ROCm/hip via rccl.rs). If multi-GPU ever moves to CUDA, NCCL + P2P wiring would be a separate effort.
