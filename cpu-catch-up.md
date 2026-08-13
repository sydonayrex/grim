# CPU backend catch-up plan

From: deficiency audit (ROCm vs CUDA/Vulkan/CPU/Metal). Grounded in `crates/grim-backend-cpu/src/lib.rs` (16-line stub) and the audit's transfer IDs T1–T7.

## 1. Current state (what CPU actually has)

- `crates/grim-backend-cpu/src/lib.rs` is a **16-line stub**: `//! CPU reference backend: host buffers, SIMD GEMM, scalar fallback routines.` That's it — no device struct, no `BackendDevice` impl visible in this file, no probe, no kernel surface, no autotuner, no cache.
- There **is** a `grim-backend-cpu` crate (it appears in the workspace), so the crate exists and presumably exposes something; but this plan is written from the stub's documented intent ("host buffers, SIMD GEMM, scalar fallback routines"), and any richer surface would need to be discovered from the crate's actual source before acting on it. This plan therefore treats the CPU backend as a **forward-reference stub** and scopes its recommendations to the stub's stated role: a reference/host-fallback path, not a tuned compute backend.

## 2. Gaps (relative to ROCm) — by design, not by accident

The CPU backend is a **reference/host-fallback** path by intent, so most of ROCm's GPU-specific strengths are not expected to transfer:
- No device discovery (no GPU to discover).
- No JIT/source compile (no GPU runtime compile concept).
- No empirical autotuner on GPU (no GPU to time).
- No HW fingerprint cache key (no device variants to fingerprint).
- No LDS/shared-memory/resource gate (no shared memory).
- No multi-GPU/collective/P2P (no GPU).
- No split-K, no TLOLog, no capability/epoch gating (no device caps).
- Broadly: the CPU backend is not the place to replicate ROCm's driver-layer architecture. That would be wrong-fit.

## 3. Transfers that DO make sense (CPU is a reference/fallback, not a GPU clone)

The transfers for CPU are about **correctness, reference, and host-side reuse**, not about making the CPU backend GPU-capable.

### T-ref-1 — CPU as the gold/reference for parity testing (highest value)
The CPU backend's real job is to be the **deterministic reference** against which GPU backends verify correctness. That means:
- CPU `BackendDevice` impls should cover the full shared op contract (matmul, add, mul, silu_mul, rms_norm, layer_norm, the backward/attention/moe variants) so every GPU op has a CPU reference that can be run on the same inputs and compared (bit-identical for F32, epsilon for FP16/BF16).
- Quant/dequant parity: the GPU quant/dequant paths need a CPU reference (dequantize → f32 → compare). The CPU backend is the natural home for the **reference dequant** of every quant format the GPU backends support (Q8_0, Q4K, Q5K, Q6K, Q3K, fp8, mxfp4, mxfp8, iq4nl/iq2s etc.) so GPU quant parity tests can compare against a known-good CPU dequant.
- This is not a "transfer from ROCm" — it's a **role** the CPU backend already is meant to fill, and it's under-filled if the stub is all there is. Concretely: verify the CPU crate actually implements the full op set needed for parity tests; if it doesn't, extend it to.

### T-ref-2 — host-side dequant reuse across backends (code reuse, not capability transfer)
CPU-hosted dequant routines are useful to **all** backends as host-side fallbacks / parity helpers. ROCm's CUDA crate already has a broad host-dequant set (q8_0, q4k, iq2/iq3/iq4nl). The pattern: a shared host-dequant module (or a CPU-backend-provided one) that every backend can call for host-side dequant/parity. If the CPU backend grows a clean, well-tested host dequant surface, the other backends can lean on it instead of duplicating per-backend host dequant methods. Direction: CPU → CUDA/Vulkan/Metal (shared host dequant), not ROCm→CPU.

### T-ref-3 — SIMD GEMM heuristics as a lightweight "autotuner analog" for CPU-only shapes
ROCm's FCP is GPU-empirical. The CPU analog is not "time on GPU" (there's no GPU) but "pick the SIMD GEMM kernel/block size that fits the CPU's vector width and L1/L2 cache for the given M/N/K." If the CPU backend ever gets a real SIMD GEMM path (the stub says "SIMD GEMM"), a **CPU-side tile/block selection** that reasons about cache lines and vector width (AVX-512/AVX2/etc.) is the CPU-native analog of ROCm's tile picker — conceptually similar (pick a tile that fits the hardware), different hardware reasoning (cache/vector width, not LDS/wavefront). Low priority until CPU GEMM is real.

### T-ref-4 — capability/format gating by CPU ISA (analog of T1, CPU-native)
Instead of ROCm's GCN-arch gating, the CPU backend can gate on the host ISA (SSE/AVX/AVX2/AVX-512, NEON on ARM) and on available instruction set features, with an epoch counter that invalidates when the running CPU differs. Pattern is the same as ROCm's `supports(mode)` — just CPU-native inputs. Only meaningful if the CPU backend has multiple SIMD kernels that depend on ISA; low priority until that exists.

## 4. What NOT to transfer to CPU (wrong-fit)

- hiprtc/JIT source parametrization — no GPU; wrong shape for a CPU backend.
- empirical GPU-time autotuner — no GPU to time; CUDA/Vulkan/Metal get the empirical subset instead.
- HW-fingerprinted JIT cache key — no device variants to fingerprint in the GPU sense.
- LDS/shared-memory/resource gate — no shared memory on CPU.
- op-identity TLOLog tile — lm_head on CPU is still a matmul; the TLOLog distinction is a GPU-tile optimization (wide-N tile for the GPU's LDS/wavefront behavior). On CPU, the same matmul just uses whatever SIMD GEMM block size fits the CPU cache — no need for a separate TLOLog tile class. Don't import the GPU tile taxonomy into CPU.
- split-K from real K — a GPU reduction-latency hiding technique; on CPU the reduction is done in the SIMD GEMM kernel's own loop structure; no need to port the GPU split-K concept.
- multi-GPU/collective/P2P — no GPUs.
- graph capture — no GPU graph concept.

## 5. Ordering

1. T-ref-1 — verify/extend CPU BackendDevice + reference dequant to cover the full op set needed for GPU parity testing (highest value; this is the CPU backend's real job).
2. T-ref-2 — if the CPU backend grows a clean host-dequant surface, make CUDA/Vulkan/Metal lean on it for host-side dequant/parity (shared, not duplicated).
3. T-ref-3/T-ref-4 — only if/when the CPU backend gets a real SIMD GEMM path with ISA-dependent kernels; then add CPU-native tile/ISA gating with an epoch counter. Deferred.

## 6. Validation

- For every GPU op (matmul, attention, moe, quant/dequant, fused ops), there is a CPU reference implementation that can be run on the same inputs and compared (bit-identical F32; epsilon for FP16/BF16).
- Every quant format the GPU backends serve has a CPU reference dequant used by parity tests.
- CPU parity tests exercise the real GPU backends against the CPU reference (the existing `standalone_quant_parity` tests on CUDA/Vulkan/Metal are the shape — CPU should be the reference they compare against).
- No GPU-specific ROCm driver-layer concept (JIT source, GPU autotuner, LDS gate, split-K, TLOLog tile, multi-GPU) is imported into the CPU backend.

## 7. Scope note

This plan treats the CPU backend as a **reference/host-fallback** path, per the stub's stated intent ("host buffers, SIMD GEMM, scalar fallback routines"). It does not attempt to make the CPU backend a tuned compute backend or to port ROCm's GPU driver-layer architecture into it — that would be wrong-fit. If the `grim-backend-cpu` crate is richer than the stub suggests, the first action is to read its actual source and revise this plan; the recommendations here are scoped to the stub until the crate's real surface is known.
