KERNEL COMPARISON: VLLM-MAIN vs GRIM-BACKEND-ROCM (and grim Vulkan for parity context)
======================================================================================

====================================================================
1. VLLM-MAIN KERNEL INVENTORY (amd/rocm surface)
====================================================================
Runtime: Triton (Python) + hand-written HIP/CU (C++) + cutlass (C++).
vLLM is multi-backend. For AMD the surface splits into:

(A) Triton AMD paths (Python, @triton.jit). File-level arch detection
    via vllm.platforms.RocmPlatform -> _GCN_ARCH. RDNA-specific tuning
    in triton kernels: smaller BLOCK, num_warps, num_stages per arch.
    Files: vllm/_aiter_ops.py (RocmPlatform.dispatch_key == "ROCM"),
          vllm/v1/attention/ops/triton_decode_attention.py (is_hip_
          branch), vllm/v1/attention/ops/turboquant_soa/* (RDNA
          warp_configs via USE_BF16_DOT/RDNA_warp_configs), vllm/kernels/
          triton/*, vllm/lora/ops/triton_ops/*. These are Pythia/LLM
          generic; they run on gfx11/gfx12 when Triton targets HIP.

(B) Hand-written HIP/CU for AMD ROCm (C++, compiled via hipcc into the
    _rocm_C extension). csrc/rocm/ is the AMD-specific C++ kernel dir.
    Files and what they are:

    csrc/rocm/attention.cu  (3717 lines)
    - paged_attention: multi-template kernel. Templated on scalar_t,
      cache_t, KV_DTYPE (auto vs fp8), OUTT, BLOCK_SIZE, HEAD_SIZE,
      NUM_THREADS, ALIBI_ENABLED, GQA_RATIO, MFMAType.
    - Two kernel families coexist in the same file: LL4MI (256-thread,
      warp-level, ~16-head-per-warp for GQA) AND a gfx11/wave32
      mfma16x16x16 variant (~64-thread, different wave layout).
    - MFMA for QK: __builtin_amdgcn_mfma_f32_4x4x4f16 (f16),
      __builtin_amdgcn_mfma_f32_16x16x16f16, and for FP8 KV cache
      __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8 / _bf8_bf8 when
      __HIP__FP8MFMA__ (gfx942/gfx950).
    - Layout engine is substantial: shared_logits[warp][4][16][4],
      shared_qk_max[warp][16], shared_exp_sum[warp][16], warpReduceMax
      via __shfl_down, two-pass exp normalization (per-warp max, then
      per-partition max), online-softmax-style rescale.
    - vLLM's AMD attention is PagedAttention + tiled softmax + MFMA.
      It IS custom AMD attention (unlike Unsloth, which delegates to
      torch.flex_attention / slow torch). This is the biggest
      structural difference vs grim's QKV path.

    csrc/rocm/q_gemm_rdna3.cu  (780 lines)
    - W4A16 GPTQ GEMM for RDNA3 (gfx1100). scalar path for M<16.
    - BLOCK_KN_SIZE=256, THREADS_X=256, each thread 4 N columns.
    - fp16 uses v_dot2_f32_f16 (__builtin_amdgcn_fdot2); bf16 widens
      to fp32 (no v_pk_fma_bf16 on gfx11); M_COUNT in {1,2,4,8}
      selected at launch by size_m.
    - Per-thread fp32 accumulator block_c[M_COUNT][4], CAS-loop
      atomic_add_pk4_f16 / atomic_add_pk4_bf16 on a 64-bit word
      (global_atomic_cmpswap_b64 retry) because gfx11 has no native
      v_global_atomic_pk_add_f16/_bf16.
    - Direct write to fp16/bf16 output (no FP32 scratch + cast pass).
    - Dequant constants per 4 columns: z1z16_h[4][2]/y1y16_h[4][2]
      (fp16 bit-trick: half2(1024+q, 1024+q*16)) or z_b_f[4]/y_b_f[4]
      (bf16 fp32 scalars). Refresh_group per group boundary.

    csrc/rocm/q_gemm_rdna3_wmma.cu  (2165 lines)
    - W4A16 WMMA for RDNA3 (gfx1100). Two kernel variants in the same
      TU:
        v1: gemm_q4_wmma_kernel_16x16_1w — 16M x 16N tile, 1 wave
            (32 threads), no __syncthreads (single-wave, s_waitcnt
            suffices).
        v2: gemm_q4_wmma_kernel_32x16_2w — 32M x 16N tile, 2 waves
            (64 threads), double-buffered LDS B-tile.
    - WMMA intrinsics: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 and
      _bf16_w32. Accumulator fp32 (v8fp32). No native 16x16x16 with
      fp16/bf16 accumulator on gfx11.
    - Wave32 register layout documented in-file (A row-major M/K in
      lane+slot, B col-major N/K, C lane=N/col, slot=M with hi-bit
      interleave; lanes 0..15 and 16..31 hold identical input frags —
      AMD "doubled" wave32 input layout; C split via lane_hi).
    - K-split heuristic: compute_wmma_k_split(size_k) -> 1/2/4 based
      on K>=1024%64==0 and K>=512%32==0; and compute_wmma_k_split_mn
      (blocks_xy vs kTargetBlocksXY=1500 on 3072-wave gfx1100) to
      avoid over-splitting.
    - Dequant "precise" variant lives HERE (not in qdq_4_rdna3.cuh):
      prep_zero_scale_fp16_precise + dequant_4bit_8_fp16_precise. fp16
      bit-trick loses ~0.025/cell at scale=0.1 with FMA form; precise
      subtracts (1024+zero) as integer first (exact in fp16 for
      integers 1024..2047) then multiplies by scale. One extra sub+mul
      per dequant pair. Also bf16->bf16 via fp32 internal (no
      v_pk_fma_bf16, avoids NaN-canonicalisation hipcc emits for
      __float2bfloat16): dequant_4bit_8_bf16_to_bf16 does widen via
      __uint_as_float(left-shift-by-16), one __fmaf_rn per element,
      then f32_to_bf16_no_canon narrow at the end.
    - Epilogue: gridDim.z>1 -> K-split atomic (pack 2 fp16/bf16 lanes
      per CAS-32 via shfl_xor pair shuffle, single CAS per even lane);
      gridDim.z==1 -> direct non-atomic write.

    csrc/rocm/moe_q_gemm_rdna3.cu  (639 lines)
    - Fused MoE W4A16 GPTQ for RDNA3. Combines expert routing
      (sorted_token_ids / expert_ids) with the RDNA3 dequant+dot from
      q_gemm_rdna3.cu.
    - Same constants: BLOCK_KN_SIZE=256, THREADS_X=256, 4 N columns,
      fp16 v_dot2 / bf16 fp32-wide dot, CAS-loop packed atomic.
    - Inputs: per-expert weight [E,K/8,N], scales [E,groups,N],
      zeros [E,groups,N/8], plus routing tensors. Launcher
      launch_moe_gemm_q4 selects BLOCK_SIZE_M (M_COUNT equivalent).
    - Epilogue applies topk_weight mul + output_topk reduction (multiple
      experts write same row via atomics).

    csrc/rocm/qdq_4_rdna3.cuh  (239 lines)
    - Shared W4A16 dequant primitives for RDNA3. fp16 path: classic
      exllamav2 bit-trick (half2(1024+q, 1024+q*16) via 0x64006400 OR;
      upper-nibble *16 works in fp16 because mantissa 10-bit holds 4-bit
      shift). bf16 path: no *16 trick (7-bit mantissa would overflow
      exponent); instead per-pair right-shift before OR with 0x43004300
      (=bf162(128,128)). bf16-input->fp32-output dequant for the scalar
      bf16 path (dequant_4bit_8_bf16_f32) and a pure-q variant
      (dequant_4bit_8_bf16_q_only) for the factored M=1 path that folds
      scale/zero outside the inner loop.

    csrc/rocm/skinny_gemms.cu, csrc/rocm/skinny_gemms_int4.cu
    - LLMM1 (matrix-vector), wvSplitK (skinny matmul with CuCount
      wave-count tuning), wvSplitKrc, wvSplitKQ (fp8).
    - wvSplitK_int4_g: W4A16 grouped skinny GEMM with per-group scales,
      optional zero_points.
    - These are the "skinny" (small-M) GEMM path; excluded on gfx1250
      (gfx9/gfx11 ISA unsupported there) via VLLM_SKIP_SKINNY_GEMMS;
      vLLM falls back to Triton/default GEMM on gfx1250 for those ops.
    - Launched via vllm._rocm_C / vllm._custom_ops.py wrappers
      (LLMM1, wvSplitK, wvSplitKrc, wvSplitKQ, wvSplitK_int4_g).

    csrc/rocm/ops.h + torch_bindings.cpp
    - torch.library op registration for _rocm_C. Registers:
      LLMM1, wvSplitK, wvSplitK_int4_g, wvSplitKrc, wvSplitKQ,
      gptq_gemm_rdna3 (dense w4a16 dispatch: scalar vs wmma based on
      size_m), gptq_gemm_rdna3_wmma (WMMA-only entry), moe_gptq_gemm_rdna3,
      paged_attention.

    csrc/rocm/attention.cu environment gating:
    - VLLM_ROCM_FP8_MFMA_PAGE_ATTN env controls mfma_type="fp8" vs "f16"
      for the paged_attention op.
    - PagedAttention op signature carries kv_cache_dtype, k_scale,
      v_scale, fp8_out_scale, mfma_type — so the same op supports
      auto (f16/bf16 KV) AND fp8 KV cache with per-tensor k/v scales
      and an optional fp8 output scale.

(C) cutlass-based AMD paths (C++). vllm uses cutlass for:
    - FP8 W8A8 block GEMM: csrc/libtorch_stable/quantization/w8a8/cutlass/
      c3x/* (sm90/sm100/sm120 fp8 scaled_mm variants, blockwise dispatch,
      azp_sm90_int8). These are CUDA SM90/100/120 cutlass paths;
      AMD uses them via the ROCm CUDA-compat layer + hipcc when the
      arch is recognized, but vLLM's primary AMD FP8 story is the
      Triton AMD FP8 paths + the csrc/rocm/attention.cu MFMA fp8 KV
      path, NOT a dedicated AMD cutlass fp8 GEMM farm the way NVIDIA
      gets.
    - W4A8 cutlass (w4a8_grouped_mm_entry, w4a8_mm_entry, w4a8_utils)
      — these are NVIDIA cutlass; AMD support here is incidental.
    - NVFP4 (fp4) cutlass sm120 kernels — NVIDIA Blackwell; AMD has
      no NVFP4 path of its own.

(D) Triton + AITER attention backends for AMD (v1). vllm.v1.attention.
    backends enumerates a priority list for AMD:
    - ROCM_ATTN (csrc/rocm/attention.cu PagedAttention via _rocm_C)
    - ROCM_AITER_FA (AITER flash-attn on cdna)
    - ROCM_AITER_UNIFIED_ATTN (AITER unified attn; rdna path when
      is_rdna_aiter_enabled)
    - ROCM_AITER_MLA (AITER MLA; gfx950-only "gluon" padding mode)
    - ROCM_AITER_MLA_SPARSE
    - TRITON_ATTN, TRITON_MLA, ROCM_AITER_TRITON_MLA, TURBOQUANT,
      FLASH_ATTN (NVIDIA cutlass flash-attn, available on AMD only
      when the CUDA-compat layer supports it), TORCH_SDPA.
    This is a richer backend selector than grim's current QKV-only
    AMD path. vLLM's AMD attention is a menu; grim's is one kernel.

(E) MLA (Multi-Query/Lightweight Attention) for AMD. vllm has:
    - vllm.v1.attention.backends.mla.rocm_aiter_mla (AITER MLA,
      gfx950 "gluon" padding mode)
    - vllm.v1.attention.backends.mla.aiter_triton_mla (AITER+Triton
      MLA)
    - vllm.v1.attention.backends.mla.triton_mla (Triton MLA)
    - vllm.v1.attention.backends.mla.flashinfer_mla, flashattn_mla
      (NVIDIA-side)
    grim has NO MLA kernel in grim-backend-rocm.

(F) Quantized KV-cache attention paths. vllm has:
    - csrc/rocm/attention.cu with KV_DTYPE==kAuto and KV_DTYPE==fp8
      (the paged_attention template supports both, with k_scale/v_scale
      and FP8_E4M3_SCALE_TARGET normalization and per-warp q_max
      reduction for fp8).
    - csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu — NVIDIA NVFP4
      KV cache; AMD has no NVFP4 KV path.
    - triton fp8 KV cache paths via vllm._aiter_ops / triton AVX.
    grim-backend-rocm has kv_dequant_attention.rs (KV-cache quantized
    attention, dequant on the fly during attention) — a different
    mechanism (dequant-on-read) vs vLLM's fp8-KV-in-the-MFMA-path.

(G) Quantization formats vLLM covers on AMD:
    - W4A16 GPTQ (csrc/rocm/q_gemm_rdna3.cu + wmma variant) — RDNA3
      only, gfx1100.
    - FP8 W8A8 block GEMM via cutlass c3x + Triton AMD FP8.
    - MXFP4 / MXFP8 via Triton (vllm.model_executor.kernels.linear.mxfp4,
      mxfp8) + flashinfer/humming/marlin/xpu backends (xpu backends are
      Intel XPU, not AMD).
    - NVFP4 (fp4) — NVIDIA Blackwell only; AMD has no NVFP4.
    - AWQ, GPTQ-marlin, marlin-int4-fp8-preprocess, compressed-tensors,
      fbgemm_fp8, torchao — mostly NVIDIA-originated; AMD support is
      via the CUDA-compat layer where applicable.
    grim covers: FP8 (standalone dequant + wmma fp8 GEMM + mxfp4/mxfp8
    standalone + fused dequant gemm fp8), MXFP4/MXFP8 standalone dequant,
    and a full K-quant/IQ-quant suite (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/IQ2/3/4).

(H) Fused MoE for AMD. vLLM has:
    - csrc/rocm/moe_q_gemm_rdna3.cu — fused W4A16 MoE for RDNA3 (fused
      routing + dequant dot, one launch per expert-token block).
    - Triton MoE kernels (vllm.model_executor.layers.fused_moe.experts/
      triton_moe.py, triton_cutlass_moe, triton_deep_gemm_moe,
      gpt_oss_triton_kernels_moe) — these run on AMD via Triton HIP.
    - AITER fused moe (rocm_aiter_ops.is_fused_moe_enabled()).
    grim has: charon.rs (sortless fused MoE dispatch, scalar + WMMA) +
      charon_wmma.rs + charon_backward.rs + scythe_persistent.rs (opcode 6
      = MoE dispatch) + comm_fuse.rs. grim's MoE is a custom sortless
      dispatch with persistent ring; vLLM's AMD MoE is fused RDNA3 W4A16
      + Triton MoE + AITER fused moe.

(I) Cross-entropy / loss, norm, rope, silu/geglu on AMD.
    - vLLM does norm/rope/silu/geglu via torch (fused_add_rms_norm op
      in csrc/libtorch_stable? check: vllm._custom_ops.fused_add_rms_norm
      is a torch.ops._C wrapper; rms_norm, fused_qk_norm_rope ops exist).
    - vLLM has csrc/libtorch_stable/fused_qknorm_rope_kernel.cu,
      fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu,
      fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu — these are fused
      QK-norm + RoPE + KV-insert kernels (CUDA/HIP). grim has none of
      these as standalone HIP kernels (norm on host/fused, RoPE on host/
      fused, silu fused into charon, no standalone norm/rope/activation
      HIP kernel in grim-backend-rocm today; vulkan HAS rms_norm.comp,
      rope.comp, silu_mul.comp).
    - vLLM cross-entropy: torch.nn.CrossEntropyLoss (no custom HIP CE
      kernel). grim has none either.

(J) Misc AMD custom ops vLLM exposes via _rocm_C:
    - ngram_compute_n_gram_ids (LongCat n-gram embedding index kernel).
    - merge_attn_states (csrc/libtorch_stable/attention/merge_attn_states.cu
      + csrc/quickreduce). grim has none of these.

====================================================================
2. VLLM VULKAN — DOES VLLM HAVE A VULKAN BACKEND?
====================================================================
No. Search for Vulkan/SPIR-V compute in vllm-main returns:
- vulkan|spirv|glsl|wgsl|compute shader|vkCreateInstance|vkCmdDispatch|
  vkEnumeratePhysicalDevices — ZERO hits in vllm-main (Python + C++).
- vllm_xpu_kernels + _xpu_ops.py — this is INTEL XPU (SYCL/oneAPI),
  not Vulkan. The file-level name "_xpu_ops" and the imports
  (vllm_xpu_kernels.flash_attn_interface) confirm it's Intel XPU.
  It includes gdn_attention (SYCL GDN kernel), fp8_gemm_w8a16,
  int4_gemm_w4a8/w4a16, fp4_gemm, deepseek_scaling_rope (XPU), etc.
  None of this is Vulkan or AMD.
- vllm's AMD story is ROCm (HIP/CU + Triton HIP + cutlass-via-hipcc),
  not Vulkan.

So vLLM has NO Vulkan backend. grim-backend-vulkan IS a real Vulkan
compute backend (lib.rs ~4482 lines, FFI to VkInstance/VkDevice/VkQueue/
VkCommandBuffer/VkShaderModule, SPIR-V shaders compiled at build time
via build.rs + glslangValidator, embedded as include_bytes!, covers
the full kernel set: norm, rope, QKV, flash, tree, paged, fused dequant
gemm for all quant formats, RWKV, selective scan, KV dequant attention,
comm_fuse, moe dispatch, etc.). This is a major asymmetry: grim has a
genuine Vulkan backend; vLLM does not.

====================================================================
3. PER-KERNEL-TYPE COMPARISON (vLLM AMD ROCm vs grim ROCm)
====================================================================

--- Paged / QKV Attention (causal, GQA) ---
vLLM: csrc/rocm/attention.cu — multi-template paged_attention. LL4MI
      (256-thread LL-style, warp-level shared_logits[warp][4][16][4],
      two-pass exp normalization, GQA scaled) AND gfx11 wave32
      mfma16x16x16 variant. MFMA for QK on f16/bf16 + optional FP8 KV
      cache with MFMA (gfx942/gfx950). PagedAttention: block_tables,
      seq_lens, query_start_loc. Multi-backend selector in v1
      (ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN,
      TURBOQUANT, FLASH_ATTN, etc.).
grim: qkv_attention.rs — causal GQA, online softmax, wave-aware
      (W64 vs W32), LDS s_max/s_sum/s_acc[8][256] per wave, Phase-1
      (no PagedAttention yet, no flash). grid=(seq_len, num_heads, 1),
      block=(256,1,1). head-dim cap at 256. cross_attention.rs for
      Whisper full cross-attention.
Comparison:
  - vLLM has a far more complete AMD attention engine: PagedAttention,
    multi-template, MFMA, FP8 KV cache support, multi-backend selector,
    MLA, NVFP4-KV (NVIDIA only), DCP (distributed kv gather).
  - grim has a simpler but hand-written causal GQA kernel (Phase-1) +
    Whisper cross-attention. grim does NOT have PagedAttention, no
    MFMA, no FP8 KV cache in the attention kernel.
  - vLLM's attention is better for production decode-with-paging;
    grim's is a research-grade Phase-1 kernel that needs PagedAttention
    + MFMA + paging to reach parity.
  - vLLM wins on attention breadth; grim's attention is narrower but
    fully owned (hand-written HIP, no Triton dependency for the QKV
    path itself).

--- RoPE ---
vLLM: fused_qknorm_rope_kernel.cu + fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
      + fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu + rotary_embedding
      op (csrc rocm? check: vllm._custom_ops.rotary_embedding wraps
      torch.ops._C.rotary_embedding — there's a HIP/CU implementation).
      DeepSeek-V4 scaling rope (XPU only via _xpu_ops).
grim: NO standalone RoPE HIP kernel in grim-backend-rocm. Vulkan HAS
      rope.comp. RoPE on ROCm is host-side or fused (not a published
      standalone HIP kernel).
Comparison:
  - vLLM has custom fused QK-norm+RoPE+KV-insert HIP kernels; grim has
    no standalone RoPE HIP kernel (host/fused only on ROCm).
  - vLLM wins on RoPE hip-kernel surface (fused variants at least).

--- RMSNorm / LayerNorm / fused QK-norm ---
vLLM: rms_norm op (torch.ops._C.rms_norm), fused_add_rms_norm op,
      fused_qknorm_rope_kernel.cu (AMD HIP), layernorm_kernels.cu /
      layernorm_quant_kernels.cu (CUDA/HIP), activation_kernels.cu.
      vLLM has a genuine norm kernel surface on AMD.
grim: NO standalone norm HIP kernel in grim-backend-rocm. norm on host
      or fused into other kernels. Vulkan HAS rms_norm.comp.
Comparison:
  - vLLM wins on AMD norm kernels; grim ROCm has none standalone.
    (grim Vulkan does.)

--- W4A16 GPTQ GEMM (dense, RDNA3) ---
vLLM: q_gemm_rdna3.cu (scalar, M<16 scalar, M>=16 dispatches to WMMA
      via gptq_gemm_rdna3 op which selects scalar vs wmma by size_m) +
      q_gemm_rdna3_wmma.cu (v1 16x16 1-wave, v2 32x16 2-wave double-
      buffered LDS). fp16 v_dot2; bf16 widens to fp32 (no v_pk_fma_bf16
      on gfx11). CAS-loop packed atomic-add (64-bit word for f16,
      32-bit word pair CAS for wmma epilogue). Precise dequant variant
      in the WMMA TU.
grim: NO W4A16 GPTQ kernel at all. grim's quantization is K-quant/IQ-
      quant + FP8/MXFP4/MXFP8, NOT GPTQ W4A16.
Comparison:
  - vLLM owns W4A16 GPTQ for RDNA3; grim has none.
  - grim owns K-quant/IQ-quant (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/IQ2/3/4);
    vLLM has none of these.
  - These are complementary quantization universes. If you want to run
    a GPTQ-w4a16-quantized Llama on RDNA3, vLLM has the path; grim does
    not. If you want Q4_K_M / Q5_K_M / IQ4_XS, grim has the path; vLLM
    does not.

--- W4A16 WMMA (RDNA3, gptq wmma) ---
vLLM: q_gemm_rdna3_wmma.cu — full WMMA engine (v1 16x16 1-wave, v2
      32x16 2-wave double-buffered). K-split heuristic (1/2/4) with
      compute_wmma_k_split_mn (blocks_xy vs 1500 target). Wave32 layout
      documented + empirically verified. Epilogue K-split atomic vs
      direct write.
grim: charon_wmma.rs — WMMA 16x16 grouped forward for Charon MoE gate/
      up/down, FP32 accum, gated behind CharonSelector/CharonVariant.
      NOT a GPTQ W4A16 WMMA; it's Charon's MoE GEMM WMMA path. Also
      wmma_gemm.rs has grim_wmma_gemm (f16 WMMA on gfx1100+ with scalar
      fallback) — a dense f16 WMMA GEMM, not GPTQ dequant.
Comparison:
  - vLLM's WMMA is W4A16 GPTQ dequant+WMMA; grim's WMMA is f16 dense
    (wmma_gemm.rs) + Charon MoE GEMM (charon_wmma.rs). Different
    problem spaces.
  - vLLM wins on RDNA3 W4A16 GPTQ WMMA; grim wins on f16 dense WMMA
    (wmma_gemm.rs) + Charon MoE WMMA.
  - If grim wants W4A16 GPTQ on RDNA3, it has no path today — that's a
    vLLM exclusive.

--- Fused MoE dispatch (AMD) ---
vLLM: moe_q_gemm_rdna3.cu — fused W4A16 MoE for RDNA3 (routing +
      dequant+dot in one launch, same scalar/WMMA split as dense,
      CAS atomic epilogue, topk_weight mul + output_topk reduction).
      Plus Triton MoE (triton_moe, triton_cutlass_moe, triton_deep_gemm_moe,
      gpt_oss_triton_kernels_moe) + AITER fused moe.
grim: charon.rs (sortless fused dispatch, gate+up SiLU+down, scalar +
      WMMA, one launch per (token,expert) pair, no host sort) + Charon
      variant system + charon_backward.rs (FP32 expert-weight backward,
      SiLU derivative) + scythe_persistent.rs (opcode 6 = MoE, persistent
      ring dispatch) + comm_fuse.rs.
Comparison:
  - vLLM's AMD fused MoE is W4A16 GPTQ + Triton MoE + AITER fused moe
    — a menu of three. grim's AMD fused MoE is Charon sortless dispatch
    + WMMA + persistent ring — one custom engine, different philosophy
    (sortless, no host sort, in-register SiLU).
  - vLLM wins on breadth (3 MoE paths on AMD); grim wins on the custom
    sortless persistent dispatch (which vLLM doesn't have — vLLM's MoE
    uses sorted_token_ids/expert_ids, i.e. host-sorted).
  - Different routing philosophies. vLLM: host-sort + fused RDNA3 kernel.
    grim: sortless device dispatch + persistent ring.

--- FP8 W8A8 block GEMM (AMD) ---
vLLM: cutlass c3x w8a8 fp8 scaled_mm (sm90/sm100/sm120 — NVIDIA names,
      AMD via hipcc CUDA-compat where recognized) + Triton AMD FP8 paths
      (vllm.kernels.triton.qkv_padded_fp8_quant, vllm.lora.ops.triton_ops
      fp8 kernel utils) + csrc/rocm/attention.cu FP8 KV cache MFMA path
      (QK MFMA on fp8 when __HIP__FP8MFMA__).
grim: fp8_standalone.rs (dequant F8->F32 per-weight), fp8_gemm_rdna4.rs
      (tiled 16x16 GEMM, float inputs, gfx1200/gfx1100/scalar — name
      historical, actually float inputs not fp8 inputs), wmma_gemm.rs
      (grim_wmma_gemm_fp8 + grim_fused_dequant_gemm_fp8 — WMMA fp8 +
      fused dequant fp8 GEMM on gfx1100+), fused_dequant_gemm.rs (generic
      f16 fused dequant+GEMM with outlier/codebook backup paths), mxfp_standalone.rs
      (MXFP4/MXFP8 dequant).
Comparison:
  - vLLM's AMD FP8 is: cutlass c3x (NVIDIA ancestry, AMD via compat) +
    Triton AMD FP8 + attention-level FP8 KV MFMA. vLLM does NOT have a
    dedicated AMD FP8 GEMM kernel in csrc/rocm/ the way it has
    q_gemm_rdna3.cu for W4A16. The FP8 attention MFMA path is KV-cache
    only.
  - grim's AMD FP8 is: standalone per-weight dequant + tiered GEMM
    (scalar -> tiled RDNA4 -> WMMA on gfx1100+) + fused dequant GEMM
    fp8 + MXFP4/8 standalone. grim has its own FP8 GEMM kernel surface
    (wmma_gemm.rs fp8 entry + fp8_gemm_rdna4.rs + fused_dequant_gemm_fp8).
  - vLLM wins on Triton+cutlass FP8 breadth; grim wins on native AMD
    HIP FP8 GEMM ownership (wmma fp8 + tiled + fused).
  - vLLM's FP8 attention MFMA path (QK on fp8) has no grim equivalent
    today (grim's qkv_attention.rs is f16/bf16, no fp8 QK MFMA).

--- MXFP4 / MXFP8 (AMD) ---
vLLM: Triton MXFP4/MXFP8 (vllm.model_executor.kernels.linear.mxfp4,
      mxfp8) + flashinfer/humming/marlin/xpu backends — xpu backends are
      Intel, not AMD. So AMD MXFP4/8 is via Triton.
grim: mxfp_standalone.rs (MXFP4 + MXFP8 dequant to F32) — native HIP
      per-weight dequant. No MXFP4/8 GEMM in grim-backend-rocm today
      (dequant only; GEMM would fuse via fused_dequant_gemm style).
Comparison:
  - vLLM: MXFP4/8 via Triton on AMD (dequant+gemm fused via Triton).
  - grim: MXFP4/8 standalone dequant (HIP), no MXFP GEMM kernel today.
  - vLLM wins on MXFP4/8 GEMM on AMD (Triton path); grim wins on native
    HIP MXFP dequant primitives.

--- NVFP4 (fp4, Blackwell) ---
vLLM: csrc/libtorch_stable/quantization/fp4/* (nvfp4_scaled_mm_kernels,
      nvfp4_scaled_mm_sm120_kernels, nvfp4_blockwise_moe_kernel,
      nvfp4_experts_quant, nvfp4_quant_entry/kernels, activation_nvfp4
      quant fusion) + vllm_xpu_kernels + _xpu_ops fp4_gemm. This is
      NVIDIA Blackwell NVFP4; AMD has no NVFP4.
grim: NO NVFP4. grim's lowest-quant is Q2_K / IQ2 family / FP8 / MXFP4.
      NVFP4 (E2M1packed) is not in grim's format set at all.
Comparison:
  - vLLM has NVFP4 for Blackwell; grim has no NVFP4. This is
    architecture-locked to SM120 / gfx12xx Blackwell. grim would need to
    add E2M1 packing + NVFP4 GEMM to enter this space.

--- K-quant / IQ-quant (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/IQ2/3/4) ---
vLLM: NONE.
grim: q4k_gemm.rs/q4k_dequant.rs, q5k_gemm.rs, q6k_gemm.rs, q2k_gemm.rs,
      q3k_gemm.rs, iq_gemm.rs/iq_dequant.rs, q8_0_dequant.rs, fused_dequant_gemm.rs
      (generic). Standalone + fused dequant+GEMM for each. Host mirror
      tests for CPU parity (dequant_q4k_grim_element_host).
Comparison:
  - grim owns the K-quant/IQ-quant universe; vLLM has zero coverage.
  - This is the single largest format-space asymmetry: grim has 7+
    GGML-style quant formats with standalone+fused HIP kernels; vLLM has
    none.

--- Cross-entropy loss ---
vLLM: torch.nn.CrossEntropyLoss (no custom HIP CE kernel).
grim: NONE (host-side).
Comparison: tie — neither has a custom HIP CE kernel. grim might want
      to port Unsloth's Triton CE kernel here (but that's Unsloth vs
      grim, not vLLM vs grim).

--- Norm / RoPE / silu/geglu as standalone kernels ---
vLLM: has rms_norm + fused_add_rms_norm + fused_qknorm_rope + activation_kernels
      (HIP/CU). grim: no standalone norm/rope/activation HIP kernel on
      ROCm (host/fused; vulkan HAS rms_norm.comp, rope.comp, silu_mul.comp).
Comparison: vLLM wins on AMD norm/rope/activation kernel surface.

--- MLA (Multi-Query/Lightweight Attention) ---
vLLM: rocm_aiter_mla + aiter_triton_mla + triton_mla backends (AMD MLA
      via AITER/Triton). grim: NONE.
Comparison: vLLM wins (has MLA; grim doesn't).

--- Selective scan (Mamba) ---
vLLM: csrc/libtorch_stable/mamba/selective_scan_fwd.cu (CUDA/HIP selective
      scan) + triton_helpers. grim: selective_scan.rs (HIP selective scan)
      + vulkan selective_scan.comp.
Comparison: both have selective scan; vLLM's is CUDA/HIP (cutlass-style
      libtorch_stable), grim's is a custom HIP kernel + Vulkan. Roughly
      tie; grim's is more DIY.

--- RWKV ---
vLLM: no dedicated RWKV kernel in the AMD surface I read (vLLM is
      Transformer-first). grim: rwkv.rs (HIP time-mix + channel-mix) +
      vulkan rwkv_time_mix/comp + rwkv_channel_mix/comp.
Comparison: grim wins — vLLM has no RWKV AMD kernel; grim has RWKV HIP
      + Vulkan.

--- KV dequant attention ---
vLLM: fp8 KV cache path in csrc/rocm/attention.cu (QK MFMA on fp8 with
      k_scale/v_scale, q_max reduction, FP8_E4M3_SCALE_TARGET). This is
      fp8-as-KV-cache, not dequant-on-read of quantized KV.
grim: kv_dequant_attention.rs (quantized KV-cache attention, dequant on
      the fly during attention) + vulkan kv_dequant_attention.comp.
Comparison: different mechanisms. vLLM: fp8 KV in MFMA. grim: dequant
      on read. Both have quantized-KV attention but via different paths.
      vLLM's is more integrated with the attention MFMA; grim's is a
      separate dequant-on-read kernel.

--- mTucker/quickreduce / merge_attn_states ---
vLLM: csrc/quickreduce + csrc/libtorch_stable/attention/merge_attn_states.cu
      (merge attention states across requests). grim: none.
Comparison: vLLM wins (has it; grim doesn't).

--- ngram embedding index kernel ---
vLLM: csrc libtorch_stable ngram kernel + vllm._custom_ops.ngram_compute_n_gram_ids.
      grim: none.
Comparison: vLLM wins.

--- Multi-GPU / comms (AMD) ---
vLLM: distributed device communicators (quick_all_reduce, all2all,
      flashinfer_all_reduce) via torch.distributed NCCL/XCCL + mori
      (gfx942/gfx950 native collectives). rocclr NCCL on AMD.
      csrc/libtorch_stable/custom_all_reduce.cu.
grim: comm_fuse.rs (SCYTHE-2 WI-6 decomposed P2P fan-in) + rccl feature
      (Cargo feature rccl) + scythe_persistent ring dispatch (opcode loop
      with all_reduce opcode). grim's comms are custom HIP + rccl; vLLM's
      are NCCL/XCCL + mori.
Comparison: both have multi-GPU comms; different mechanisms. vLLM's is
      NCCL/mori (standard); grim's is custom SCYTHE comm_fuse + rccl
      feature. vLLM's is more standard/integrated; grim's is custom and
      tied to the Scythe persistent model.

--- Triton AMD attention backends (v1) ---
vLLM has a whole menu:
  ROCM_ATTN (csrc rocm paged attn) > ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN
  > TRITON_ATTN > TURBOQUANT > FLASH_ATTN > TORCH_SDPA, plus MLA variants.
  This is a rich, prioritized, validated-per-configuration attention
  backend selector for AMD.
grim: single QKV attention kernel (qkv_attention.rs) + cross_attention.rs.
      No backend selector; no Triton attention, no AITER, no flash, no
      turboquant, no MLA.
Comparison: vLLM wins decisively on attention backend breadth and
      selection logic.

--- Vulkan ---
vLLM: NONE. vllm_xpu_kernels/_xpu_ops is Intel XPU (SYCL), not Vulkan.
      No SPIR-V, no GLSL, no Vulkan compute backend in vllm-main.
grim: grim-backend-vulkan IS a real Vulkan compute backend (~4482 lines
      lib.rs, VkInstance/Device/Queue/CommandBuffer/ShaderModule FFI,
      SPIR-V shaders compiled at build time + embedded as include_bytes!,
      full kernel set coverage: norm/rope/QKV/flash/tree/paged/fused-
      dequant-gemm-all-formats/RWKV/selective-scan/KV-dequant-attention/
      comm_fuse/moe-dispatch/etc.).
Comparison: grim has a genuine Vulkan backend; vLLM has none. This is a
      major asymmetry — if you want LLM inference on a Vulkan device
      ( integrated GPU, non-ROCm AMD, MoltenVK-on-macOS, etc.), grim has
      a path; vLLM does not.

====================================================================
4. SUMMARY TABLE (kernel type x repo)
====================================================================
Legend:  Y = has custom kernel  ~ = via Triton on AMD (no native AMD HIP
         source in csrc/rocm/)  N = none  NV = NVIDIA-only (cutlass/NVFP4)
         G = grim Vulkan (not ROCm)  - = N/A

Kernel Type                       | vLLM AMD ROCm         | grim ROCm (HIP)
----------------------------------|-----------------------|----------------
PagedAttention (causal GQA)      | Y (csrc/rocm/attention.cu, LL4MI+mfma16 wave32, multi-template, FP8 KV MFMA) | N (Phase-1 QKV only, no paging)
Flash / turboquant attention     | Y (FLASH_ATTN via compat + TURBOQUANT Triton) | N
MLA (Multi-Query/Lightweight)    | Y (rocm_aiter_mla, aiter_triton_mla, triton_mla) | N
Cross-attention (Whisper-style)  | N (not in AMD surface read) | Y (cross_attention.rs)
RoPE (standalone/fused)          | Y (fused_qknorm_rope_kernel.cu + rotary_embedding op + deepseek scaling rope on XPU only) | N (host/fused; G: rope.comp)
RMSNorm / fused QK-norm          | Y (rms_norm op, fused_add_rms_norm, fused_qknorm_rope_kernel.cu, layernorm_kernels.cu) | N (host/fused; G: rms_norm.comp)
W4A16 GPTQ GEMM (dense, RDNA3)   | Y (q_gemm_rdna3.cu scalar + q_gemm_rdna3_wmma.cu v1/v2 WMMA) | N (not in grim's format universe)
W4A16 GPTQ WMMA (RDNA3)          | Y (q_gemm_rdna3_wmma.cu) | N (wmma_gemm.rs is f16 dense, not GPTQ dequant)
Fused MoE (W4A16 GPTQ, RDNA3)   | Y (moe_q_gemm_rdna3.cu) | N (not W4A16)
Fused MoE (sortless custom dispatch) | N (vLLM MoE uses host-sorted) | Y (charon.rs + scythe_persistent opcode 6)
FP8 W8A8 block GEMM (AMD native) | ~ (cutlass c3x via compat + Triton AMD FP8; no dedicated AMD csrc/rocm FP8 GEMM) | Y (wmma_gemm.rs fp8 + fp8_gemm_rdna4.rs + fused_dequant_gemm_fp8 + fp8_standalone.rs)
FP8 KV-cache MFMA attention       | Y (csrc/rocm/attention.cu MFMA fp8 QK path) | N (qkv_attention.rs is f16/bf16; no fp8 QK MFMA)
MXFP4 / MXFP8 dequant+gemm      | ~ (Triton MXFP4/8 on AMD) | Y (mxfp_standalone.rs dequant only; no MXFP GEMM today)
NVFP4 (fp4, Blackwell)           | Y (csrc/libtorch_stable/quantization/fp4/*, NVFP4) | N (format not in grim's set)
K-quant / IQ-quant (Q4_K..Q6_K, IQ2/3/4) | N (zero coverage) | Y (q4k/q5k/q6k/q2k/q3k/iq_gemm + standalone dequants + fused)
Q8_0 dequant/GEMM               | N (zero coverage) | Y (q8_0_dequant.rs + fused path)
MoE grouped GEMM (Triton)        | ~ (triton_moe, triton_cutlass_moe, triton_deep_gemm_moe on AMD) | N (different approach: Charon sortless + ring)
MoE backward (expert weights)    | N visible in AMD surface read (Triton MoE may have backward, not in csrc/rocm) | Y (charon_backward.rs, FP32)
Persistent dispatch ring         | N | Y (scythe_persistent.rs, opcode 6 = MoE)
Elementwise (add/mul/sqrt/recip) | N (PyTorch/torch ops) | N (inlined in fused; G: standalone comps)
Softmax (standalone)             | N (inside attention) | N (inside qkv_attention online softmax; G: softmax.comp)
Embedding (standalone)           | N | N (G: embedding.comp)
RWKV time/channel mix            | N | Y (rwkv.rs + G: rwkv_*.comp)
Selective scan (Mamba)           | ~ (csrc/libtorch_stable/mamba/selective_scan_fwd.cu CUDA/HIP + triton_helpers) | Y (selective_scan.rs + G: selective_scan.comp)
KV dequant attention             | Y (fp8 KV cache MFMA path in attention.cu) | Y (kv_dequant_attention.rs + G: kv_dequant_attention.comp) [different mechanism]
Quickreduce / merge_attn_states  | Y (csrc/quickreduce + merge_attn_states.cu) | N
ngram embedding index kernel     | Y (ngram kernel + _custom_ops.ngram_compute_n_gram_ids) | N
CommFuse / AllReduce (multi-GPU) | ~ (NCCL/XCCL + mori gfx942/950 + custom_all_reduce.cu) | Y (comm_fuse.rs + rccl feature + scythe ring all_reduce opcode)
Vulkan compute backend           | N (vllm_xpu_kernels is Intel XPU SYCL, not Vulkan) | Y (grim-backend-vulkan, full SPIR-V backend)

====================================================================
5. KEY TAKEAWAYS
====================================================================

1. VLLM AMD ATTENTION IS PRODUCTION-GRADE; GRIM'S IS PHASE-1
   vLLM's csrc/rocm/attention.cu is a multi-template PagedAttention with
   LL4MI (256-thread LL-style) + gfx11 wave32 mfma16x16x16 variants,
   MFMA for QK on f16/bf16 + optional FP8 KV cache MFMA, two-pass exp
   normalization, GQA scaled, paging. Plus a rich v1 backend selector
   (ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN,
   TURBOQUANT, FLASH_ATTN, TORCH_SDPA, MLA variants). grim's
   qkv_attention.rs is a Phase-1 causal GQA kernel with online softmax,
   wave-aware (W64/W32), no paging, no MFMA, no FP8 KV, no multi-backend
   selector. vLLM wins on attention by a wide margin.

2. VLLM OWNS W4A16 GPTQ ON RDNA3; GRIM DOESN'T
   vLLM's q_gemm_rdna3.cu (scalar) + q_gemm_rdna3_wmma.cu (v1 16x16
   1-wave, v2 32x16 2-wave double-buffered WMMA) + moe_q_gemm_rdna3.cu
   (fused MoE W4A16) is a complete W4A16 GPTQ engine for RDNA3 gfx1100.
   grim has NO W4A16 GPTQ path at all — grim's quantization universe is
   K-quant/IQ-quant + FP8/MXFP4/MXFP8. These are different worlds. If you
   want to run a GPTQ-w4a16 quantized model on RDNA3, vLLM has the path;
   grim does not.

3. GRIM OWNS K-QUANT/IQ-QUANT; VLLM HAS ZERO
   The single largest format-space asymmetry: grim has Q4_K/Q5_K/Q6_K/
   Q2_K/Q3_K/IQ2_XXS/XS/S/IQ3_XXS/S/IQ4_NL/XS dequant + fused GEMM (7+
   formats, standalone + fused, fwd + bwd for IQ). vLLM has zero K-quant/
   IQ-quant coverage. If you want to run a Q4_K_M / IQ4_XS model on ROCm,
   grim has the path; vLLM does not.

4. VLLM HAS MORE FORMAT BREADTH; GRIM HAS MORE NATIVE HIP OWNERSHIP
   vLLM's AMD quantization is: W4A16 GPTQ (native HIP csrc/rocm) + FP8
   (cutlass c3x via compat + Triton AMD + attention FP8 KV MFMA) + MXFP4/
   MXFP8 (Triton) + NVFP4 (NVIDIA only) + AWQ/GPTQ-marlin/compressed-
   tensors/fbgemm/torchao (mostly NVIDIA, AMD via compat). grim's AMD
   quantization is: FP8 (native HIP standalone dequant + wmma fp8 GEMM +
   tiled fp8 GEMM + fused dequant fp8 GEMM) + MXFP4/8 (native HIP
   standalone dequant) + K-quant/IQ-quant (native HIP) + Q8_0 (native
   HIP). vLLM has more formats; grim has more formats implemented as
   native AMD HIP kernels it fully owns (vs cutlass-via-compat or Triton).

5. VLLM HAS MLA, NORM/rope FUSION, QUICKREDUCE, NGRAM — GRIM DOESN'T
   vLLM's AMD surface includes: fused_qknorm_rope_kernel.cu (QK-norm +
   RoPE + KV-insert), fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
   (DeepSeek-V4), fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu
   (Minimax-M3), rms_norm + fused_add_rms_norm ops, quickreduce +
   merge_attn_states.cu, ngram_compute_n_gram_ids kernel. grim has none
   of these as standalone HIP kernels (norm/rope/silu on host/fused in
   grim; vulkan HAS rms_norm.comp, rope.comp, silu_mul.comp).

6. VLLM HAS NO VULKAN BACKEND; GRIM DOES
   vllm_xpu_kernels + _xpu_ops.py is INTEL XPU (SYCL/oneAPI), not Vulkan.
   There is no SPIR-V, no GLSL, no Vulkan compute backend in vllm-main.
   grim-backend-vulkan IS a real Vulkan compute backend (~4482 lines,
   VkInstance/Device/Queue/CommandBuffer/ShaderModule FFI, SPIR-V shaders
   compiled at build time + embedded as include_bytes!, full kernel set).
   If you want LLM inference on a Vulkan device, grim has a path; vLLM
   does not.

7. VLLM MoE = HOST-SORT + FUSED RDNA3 / TRITON / AITER; GRIM MoE =
   SORTLESS DEVICE DISPATCH + PERSISTENT RING
   vLLM's AMD MoE is: csrc/rocm/moe_q_gemm_rdna3.cu (fused W4A16 MoE,
   host-sorted token_ids/expert_ids) + Triton MoE (triton_moe,
   triton_cutlass_moe, triton_deep_gemm_moe, gpt_oss_triton_kernels_moe)
   + AITER fused moe. grim's AMD MoE is: charon.rs (sortless fused
   dispatch, gate+up SiLU+down, scalar+WMMA, one launch per (token,expert)
   pair, no host sort) + charon_wmma.rs + charon_backward.rs (FP32 expert
   weight backward, SiLU derivative) + scythe_persistent.rs (opcode 6 =
   MoE, persistent ring dispatch) + comm_fuse.rs. Different philosophies:
   vLLM host-sorts then fires fused kernels; grim sorts inside the device
   via a sortless dispatcher + persistent ring.

8. GRIM'S ROCM ATTENTION IS NARROWER BUT FULLY OWNED; VLLM'S IS A MENU
   vLLM AMD attention = menu of backends (ROCM_ATTN + AITER + Triton +
   turboquant + flash + torch SDPA + MLA). grim ROCm attention = one
   hand-written HIP kernel (qkv_attention.rs) + cross_attention.rs. vLLM
   wins on selection/breadth; grim wins on "we wrote the kernel ourselves
   in HIP, no Triton/AITER dependency for the QKV path." But grim's single
   kernel is Phase-1 — it needs paging + MFMA + FP8 KV + multi-backend
   selection to reach vLLM's production breadth.

9. VLLM FP8 ATTENTION MFMA HAS NO GRIM EQUIVALENT
   vLLM's csrc/rocm/attention.cu has a QK MFMA path on FP8 when
   __HIP__FP8MFMA__ (gfx942/gfx950) with k_scale/v_scale, q_max
   reduction, FP8_E4M3_SCALE_TARGET=224 normalization. grim's
   qkv_attention.rs is f16/bf16 only — no fp8 QK MFMA. If grim wants
   fp8-KV attention on CDNA, that's a vLLM-exclusive feature today.

10. VLLM SKINNY GEMM (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) IS VLLM-ONLY
    vLLM has csrc/rocm/skinny_gemms.cu + skinny_gemms_int4.cu (LLMM1,
    wvSplitK, wvSplitKrc, wvSplitKQ, wvSplitK_int4_g) — matrix-vector +
    skinny matmul with CuCount wave-count tuning, excluded on gfx1250.
    grim has NO skinny GEMM path. grim's GEMM is dense tiled/WMMA/fused
    dequant. If you need matrix-vector or skinny matmul on AMD, vLLM has
    it; grim doesn't.

====================================================================
6. BOTTOM LINE
====================================================================
vLLM's AMD ROCm kernel surface is broader and more production-oriented:
PagedAttention + MFMA + FP8 KV + W4A16 GPTQ (scalar+WMMA) + fused W4A16
MoE + norm/rope fusion + MLA + NVFP4 + quickreduce + ngram + a rich
attention backend selector. But vLLM's AMD depth is concentrated in
W4A16-GPTQ (gfx1100 only), FP8 (cutlass-via-compat + Triton), and
attention — and vLLM has NO K-quant/IQ-quant, NO native AMD FP8 GEMM farm
(dedicated csrc/rocm FP8 GEMM), NO Vulkan backend, NO RWKV, and its MoE
is host-sorted (no sortless device dispatch).

grim's AMD ROCm kernel surface is narrower but deeper in the formats it
owns: K-quant/IQ-quant (7+ formats, standalone+fused, fwd+bwd), native
HIP FP8 GEMM (wmma fp8 + tiled + fused dequant), native HIP MXFP4/8
dequant, custom sortless MoE dispatch (Charon + WMMA + persistent ring),
custom causal GQA QKV attention (Phase-1, no paging) + Whisper cross-
attention, RWKV, selective scan, KV dequant attention, comm_fuse/rccl.
grim has NO W4A16 GPTQ, NO PagedAttention, NO MFMA, NO FP8 KV attention,
NO MLA, NO norm/rope standalone HIP kernels (on ROCm; vulkan has them),
NO NVFP4, NO Vulkan-in-vLLM-equivalent competition (vLLM has no Vulkan at
all).

These are largely complementary:
- vLLM wins on: attention production breadth (PagedAttention+MFMA+FP8 KV+
  backend selector+MLA), W4A16 GPTQ on RDNA3, norm/rope fusion, NVFP4,
  quickreduce/merge_attn_states, ngram, multi-GPU via NCCL/mori.
- grim wins on: K-quant/IQ-quant formats, native HIP FP8 GEMM ownership,
  native HIP MXFP dequant, sortless MoE dispatch + persistent ring, RWKV,
  custom QKV attention (owned, no Triton), Vulkan backend (vLLM has none),
  KV dequant attention (own mechanism).
- The one area where they're directly comparable and vLLM clearly wins:
  AMD attention (PagedAttention+MFMA+FP8 KV+selector vs grim Phase-1
  QKV). If grim wants to close that gap, it needs to add PagedAttention,
  MFMA for QK, FP8 KV cache support, and ideally a multi-backend attention
  selector — or port vLLM's csrc/rocm/attention.cu approach (with the
  caveats that vLLM's is C++/hipcc, not Rust, and is gated on gfx9/gfx11
  with the fp8 MFMA path gated on gfx942/gfx950).
- The one area where they're directly comparable and grim clearly wins:
  K-quant/IQ-quant (vLLM has none) + Vulkan backend (vLLM has none) +
  sortless MoE dispatch (vLLM host-sorts).

Report done. Archives: vllm-main csrc/rocm/ (attention.cu, q_gemm_rdna3.cu,
q_gemm_rdna3_wmma.cu, moe_q_gemm_rdna3.cu, qdq_4_rdna3.cuh, torch_bindings.cpp,
skinny_gemms.cu, skinny_gemms_int4.cu, ops.h), vllm _custom_ops.py + _xpu_ops.py
+ platforms/rocm.py + v1 attention backends, vllm quantization kernel dirs,
vllm Triton kernel dirs, vllm libtorch_stable quantization/attention/moe dirs,
grim-backend-rocm kernel sources re-read, grim-backend-vulkan lib.rs (Vulkan
parity check). No external sources used.