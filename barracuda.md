# Engineering Plan: Full CUDA Parity with ROCm Backend

This plan addresses all 4 major parity gaps between `grim-backend-cuda` and `grim-backend-rocm`:

---

## 1. MoE Kernel Specialization & Charon Parity
**Objective:** Replace standard sequential/un-fused MoE dispatch with the full **Charon** architecture for CUDA.

- **`crates/grim-backend-cuda/src/kernels/charon.rs`**:
  - Implement CUDA C++ / PTX kernel `grim_moe_fused_dispatch` matching the sortless fused dispatch GEMM.
  - Implement token-sorted grouped fused MoE (`grim_moe_fused_grouped`): Gate + Up fused SiLU combine + Down projection with atomic accumulation into shared token slots.
- **`crates/grim-backend-cuda/src/kernels/charon_wmma.rs`**:
  - Tensor Core (WMMA / `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`) grouped GEMM tiles for MoE layers.
- **`crates/grim-backend-cuda/src/kernels/charon_backward.rs`**:
  - Implement backward pass: `d_gate_w`, `d_up_w`, `d_down_w`, and `d_x` computation on GPU.
- **`crates/grim-backend-cuda/src/kernels/moe_mega_kernel.rs`**:
  - Persistent cooperative thread array (CTA) scheduler for multi-expert top-k execution.

---

## 2. Quantized GEMM Coverage (IQ / K-Quants / MXFP4)
**Objective:** Add native fused dequant-GEMM compute kernels matching ROCm's library.

- **`crates/grim-backend-cuda/src/kernels/mxfp4_gemm.rs` & `mxfp_standalone.rs`**:
  - Direct microscale FP4 (E2M1) and FP8 (E4M3) Tensor Core GEMM kernels using warp-level scale expansion.
- **`crates/grim-backend-cuda/src/kernels/iq_gemm.rs` & `iq_dequant.rs`**:
  - Implement native CUDA kernels for `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`.
- **`crates/grim-backend-cuda/src/kernels/q_gemm/`**:
  - `q2k_gemm.rs`, `q3k_gemm.rs`, `q4k_gemm.rs`, `q5k_gemm.rs`, `q6k_gemm.rs` with shared-memory double-buffering.
- **`crates/grim-backend-cuda/src/kernels/compressed_gemm.rs` & `marlin_gemm.rs`**:
  - Marlin-format and AWQ W4A16 / W8A8 compute kernels.

---

## 3. Distributed Training & Collectives (NCCL & FSDP)
**Objective:** Multi-GPU parameter sharding, ZeRO-3, and direct VRAM collective execution.

- **`crates/grim-backend-cuda/src/nccl.rs`**:
  - Dynamic loading of `libnccl.so.2` / `libnccl.so` using `libloading` with graceful fallback.
  - Implement `ncclAllReduce`, `ncclReduceScatter`, `ncclAllGather`, `ncclSend`, `ncclRecv`, and `ncclGroupStart`/`ncclGroupEnd`.
- **`crates/grim-backend-cuda/src/fsdp.rs`**:
  - Port `ConsumerFsdpGroup`, `ConsumerFsdpConfig`, parameter slicing, and collective gradient accumulation.
- **`crates/grim-backend-cuda/src/device/parallel_comm.rs`**:
  - Bridge NCCL communicators and host ring staging to `CollectiveOps` on `CudaDevice`.

---

## 4. Advanced Attention Architectures & Speculative Sampling
**Objective:** FlashAttention, FlashDecode, Multi-Head Latent Attention (MLA), and speculative decoding.

- **`crates/grim-backend-cuda/src/kernels/flash_decode.rs`**:
  - FlashDecode split-KV persistent reduction kernel for long context decode.
- **`crates/grim-backend-cuda/src/kernels/mla_decode.rs`**:
  - DeepSeek MLA low-rank key-value projection and latent cache kernel.
- **`crates/grim-backend-cuda/src/kernels/sage_attention.rs` & `mrope.rs`**:
  - SageAttention INT8/FP8 quantized attention and Multi-dimensional RoPE (Qwen-VL).
- **`crates/grim-backend-cuda/src/kernels/speculative_sampler.rs`**:
  - On-device draft token verification and tree speculative sampling.

---

## Phased Rollout Order
1. **Phase 1**: Distributed Training & Collectives (`src/nccl.rs`, `src/fsdp.rs`, `src/device/parallel_comm.rs`).
2. **Phase 2**: Charon MoE Engine (`charon.rs`, `charon_wmma.rs`, `moe_mega_kernel.rs`).
3. **Phase 3**: Quantized GEMM Expansion (IQ / K-quants / MXFP4 Tensor Core kernels).
4. **Phase 4**: Advanced Attention & Speculative Sampling (FlashDecode, MLA, Speculative Sampler).
