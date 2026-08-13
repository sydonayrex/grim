GRIM-TO-VLLM PORT DOC: SECTIONS 1-4 VERDICT
==============================================

This verdict checks the doc's claims about grim's current state against
the actual source files in crates/grim-backend-rocm (and subdirs). Files
read for this verdict: qkv_attention.rs (full), compute_kernels.rs (full),
source_asm.rs (full), gptq_kernel.rs (found at src/gptq_kernel.rs, NOT in
kernels/), wmma_gemm.rs (full), device_cubecl.rs (gptq_correction fn),
lib.rs (gptq_kernel mod), device_cubecl.rs test (gptq_correction test),
plus content searches: "gptq" (24 matches across crate), "skinny|LLMM1|
wvSplit|wvmSplit" in kernels/ (0 matches), "matrix.vector|vector.product"
in kernels/ (0 matches).

--------------------------------------------------------------------------------
SECTION 1 — PagedAttention / QK MFMA attention
--------------------------------------------------------------------------------

DOC CLAIM:
  "STUB paged path (BlockTableEntry, page walk) -- exists in source but
   is incomplete vs vLLM's real paged attention."
  "No MFMA. No FP8 KV."
  "grim_qkv_attention_paged exists but is incomplete."

VERIFIED STATE:

  qkv_attention.rs:
    - grim_qkv_attention_paged (lines 208-518): REAL kernel. Has:
        * BlockTableEntry struct (line 203-206, also line 522-526 in Rust).
        * Full page-walk math: b = j / page_size, t = j % page_size,
          physical_token_idx = entry.block_id * page_size + t (lines 281-284).
        * Causal guard: if (j > abs_i || j >= kv_seq_len) break (line 278).
        * Online softmax with wave merge in LDS s_max/s_sum/s_acc[8][256],
          same wave-merge pattern as non-paged kernel (lines 256-356).
        * GQA mapping: q_per_kv, kv_head (lines 227-228).
        * Q layout [batch, num_heads, head_dim], K/V pages as
          [num_pages, page_size, num_kv_heads, head_dim] (lines 231, 286).
        * Rust launcher launch_paged_attention (lines 534-600+): takes
          q, block_tables, k_pages, v_pages as RocmStorage, grid=(batch,
          num_heads, 1), block=(wf*4, 1, 1), wavefront-aware (W32->128,
          W64->256 threads).
      This is a COMPLETE paged attention kernel, not a stub. The doc's
      "STUB...incomplete" claim is FALSE.

    - MFMA: NOT in qkv_attention.rs. No __builtin_amdgcn_mfma* anywhere in
      this file (the file is 726 lines, the HIP literal ends at line 519,
      the rest is Rust launcher). So "no MFMA in the attention kernel path"
      is TRUE.

    - BUT: wmma_gemm.rs (read in full) has a REAL gated MFMA scaffold:
        * Lines 283-363: "#if defined(__gfx1200__) || defined(__gfx1201__)"
          MFMA gates. grim_fused_dequant_gemm_fp8_mfma (lines 305-339) and
          grim_fused_dequant_backward_gemm_fp8_mfma (lines 342-361).
        * The MFMA kernel has a placeholder instruction: line 301
          "__asm__ volatile(\"\" : : \"r\"(packed)); // placeholder — actual
          MFMA uses vcvt_f32_f8". Lines 329-332: "gfx1200 mfma_f32_32x32x32
          _f8 equivalent — scalar fallback here. On real CDNA hardware, these
          32 FP8→F32 conversions happen via the mfma instruction itself. This
          scalar fallback ensures compilation on non-gfx1200 targets."
        * So the MFMA scaffold is REAL and HONESTLY LABELED. It's not a real
          MFMA instruction (it's a scalar fallback in a placeholder), but the
          gate structure, kernel entry, and documentation are real. A real
          MFMA instruction can be slotted in where the placeholder is.
      So "no MFMA at all" understates the gap. There's a real scaffold.

    - FP8 KV: NOT in qkv_attention.rs. TRUE gap. No k_scale/v_scale FP8 KV
      path.

VERDICT:
  - "STUB...incomplete" for paged: FALSE. Paged kernel is complete. Correct
    the doc: "grim_qkv_attention_paged is a complete paged-attention kernel
    (BlockTableEntry, page-walk math, causal guard, online softmax, GQA,
    wave-aware launch). The gap vs vLLM is feature BREADTH (vLLM's is multi-
    template with MFMA + FP8 KV + ALIBI + GQA ratio variants), not presence
    of paging."
  - "No MFMA": TRUE that there's no MFMA instruction in the attention kernel,
    but the doc should note that wmma_gemm.rs HAS a real gated MFMA scaffold
    (gfx1200/1201 gate, placeholder instruction, honestly labeled) for the
    fused-dequant-FP8 path. The gap is "no MFMA in attention" + "MFMA scaffold
    exists elsewhere but uses a placeholder." Correct to: "No MFMA instruction
    in the attention kernel yet. A real gated MFMA scaffold exists in
    wmma_gemm.rs (gfx1200/1201 gate, placeholder instruction that compiles on
    non-gfx1200 targets) — slotting a real mfma instruction into the attention
    path for gfx1100+ (RDNA3) is part of the port."
  - "No FP8 KV": TRUE.

--------------------------------------------------------------------------------
SECTION 2 — W4A16 GPTQ GEMM (dense + WMMA, RDNA3)
--------------------------------------------------------------------------------

DOC CLAIM:
  "NO W4A16 GPTQ at all."
  "grim's quantization universe is K-quant/IQ-quant + FP8/MXFP4/MXFP8."

VERIFIED STATE:

  gptq_kernel.rs EXISTS at crates/grim-backend-rocm/src/gptq_kernel.rs
  (NOT in kernels/ subdirectory — it's at src/gptq_kernel.rs). Confirmed via:
    - lib.rs line 31: "pub mod gptq_kernel;"
    - lib.rs line 107: "pub use crate::gptq_kernel::wavefront_size_for_gcn;"
    - Content search "gptq" across crate: 24 matches, including:
        * src/gptq_kernel.rs (286 lines)
        * src/lib.rs (mod decl + re-export)
        * tests/device_cubecl.rs (gptq_correction test)
        * src/device/cubecl.rs (gptq_correction fn)
      AND NO matches in any file under src/kernels/.

  What gptq_kernel.rs implements (READ IN FULL, lines 1-286):
    - GPTQ_CORRECTION_KERNEL (lines 24-59): "gptq_wavefront_correction_kernel".
      This implements the GPTQ error-correcting update:
        W_corrected = W_approx + α * H_diag^{-1} ⊙ (W_original - W_approx)
      Each HIP thread corrects one element using the diagonal Fisher
      preconditioner. This is OFFLINE WEIGHT CORRECTION (re-quantization
      calibration pass), NOT inference-time dequant+GEMM.
      Doc comment (lines 1-14) explicitly says: "ROCm HIP kernels for GPTQ
      quantization-aware RE-QUANTIZATION. Provides wavefront-level parallelism
      for the GPTQ error-correcting update." Line 9-11: "This is the Pass 4
      ROCm-accelerated path: the CPU fallback in grim-quant runs scalar row-
      by-row; this module runs the same algorithm with wavefront-parallel
      HIP kernels."

    - GPTQ_SCALE_FIT_KERNEL (lines 66-262+): "gptq_scale_fit_kernel".
      GPU-accelerated per-block scale search. One HIP thread per quantization
      block, evaluates all 7 scale multipliers (0.6, 0.75, 0.9, 1.0, 1.1,
      1.25, 1.4), picks the one with lowest weighted quantization error.
      This is SCALE SEARCH (finding the best per-block scale), NOT inference-
      time dequant+GEMM.

    - So gptq_kernel.rs implements TWO THINGS, both OFFLINE (calibration/
      re-quantization), NOT inference-time:
        1. Wavefront correction (Hessian-diagonal Fisher update).
        2. Scale fit (per-block scale search across 7 multipliers).

    These are real, useful, and correctly named FOR WHAT THEY ARE. But they
    are NOT the w4a16 dequant+GEMM inference path that section 2 is about.

  What section 2 is about (inference-time w4a16 dequant+GEMM):
    - W4A16 codes (nibble-per-byte, GPTQ bit-trick) -> dequantized FP16/BF16
      weights at inference time, then GEMM with those weights.
    - vLLM's q_gemm_rdna3.cu (scalar) + q_gemm_rdna3_wmma.cu (WMMA) +
      moe_q_gemm_rdna3.cu (fused MoE) implement this.
    - grim has NO w4a16 inference dequant+GEMM kernel (confirmed: no
      "w4a16" or "half2(1024" or GPTQ dequant bit-trick in any kernel file).

  NAMING COLLISION CONFIRMED:
    - A user searching "gptq" in the crate WILL find gptq_kernel.rs and see
      "GPTQ HIP kernels" and may wrongly conclude "GPTQ is implemented."
    - But what's implemented is OFFLINE calibration (correction + scale fit),
      NOT inference-time w4a16 dequant+GEMM.
    - The naming overlap is a real risk. gptq_kernel.rs is correctly named
      for what it does (GPTQ re-quantization kernels), but the term "GPTQ"
      spans both the calibration step AND the inference-time format. The doc
      should warn about this.

  W4A16 inference path: NOT found in any kernel file. Confirmed by:
    - Content search "w4a16" across crate: no matches in any .rs file.
    - Content search "half2(1024" (the GPTQ bit-trick signature): no matches.
    - Content search "q_gemm_rdna3" or "gptq_gemm": no matches.
    - The kernels dir has wmma_gemm (f16 WMMA, not GPTQ), fused_dequant_gemm
      (generic, not GPTQ), fp8_gemm_rdna4 (FP8, not W4A16), and the full
      K-quant/IQ-quant suite — but NO w4a16 inference GEMM.

VERDICT:
  - "NO W4A16 GPTQ at all" for INFERENCE: TRUE. No w4a16 inference dequant+
    GEMM kernel exists. The doc's claim is correct for what section 2 is
    about.
  - BUT the doc's phrasing "NO W4A16 GPTQ at all" is too absolute — it
    ignores that gptq_kernel.rs EXISTS and implements GPTQ-related kernels
    (offline correction + scale fit). The doc should be more precise: "No
    INFERENCE-TIME w4a16 dequant+GEMM kernel. Note: src/gptq_kernel.rs
    implements OFFLINE GPTQ calibration (wavefront correction + scale fit),
    which is a different thing from the inference path in this section — the
    naming overlap risks confusion."
  - NAMING COLLISION: CONFIRMED and REAL. The doc should warn: "If a file
    named gptq_kernel.rs exists (it does, at src/gptq_kernel.rs), it
    implements offline GPTQ calibration (Hessian-diagonal weight correction +
    scale search), NOT inference-time w4a16 dequant+GEMM. Searching 'gptq'
    in the crate will surface this file and may wrongly suggest section 2 is
    in progress or complete. When porting section 2, use clear naming that
    distinguishes inference-time dequant+GEMM (e.g. w4a16_inference_gemm)
    from offline calibration (gptq_kernel)."

--------------------------------------------------------------------------------
SECTION 3 — Fused QK-norm + RoPE + KV-insert kernels
--------------------------------------------------------------------------------

DOC CLAIM:
  "NO standalone norm/rope/activation HIP kernels on ROCm."
  "grim's ROCm side has no norm or rope HIP kernel."
  "norm on host/fused, RoPE on host/fused, silu fused into charon."

VERIFIED STATE (compute_kernels.rs, full file read):

  The file contains MANY real HIP kernels in the OTHER_KERNEL_SOURCE literal
  (lines 4-391). Confirmed real HIP kernels:

    * grim_rope (lines 35-62): standalone plain full-rotary RoPE. "One thread
      per (batch, step, dim-half-pair) element." 3-D input [B,S,D],
      positions[si] per step, rotation to pairs (x[i], x[half+i]).
      CONTRACT: rotary_dim == d (full rotary). Use grim_rope_yarn for
      partial/YaRN. This is a STANDALONE RoPE HIP kernel on ROCm. NOT host.

    * grim_rope_yarn (lines 79-123): standalone partial-rotary + YaRN kernel.
      "Handles rotary_dim <= d (partial) and pre-computed YaRN-ramp
      frequencies." inv_freq param, mscale param, rotary_half param.
      Two-pass: rotate [0, rotary_half) pairs, copy non-rotary dims
      [2*rotary_half, d) verbatim. STANDALONE partial/YaRN HIP kernel on ROCm.

    * grim_rms_norm (lines 192-208): standalone RMSNorm HIP kernel. "Computes
      variance = sum(x*x)/row_len, rms = sqrt(variance + eps), out = x *
      w[col] / rms." Fixed a bug (lines 203-207 comment): prior code indexed
      w by global linear index (garbage for every row past the first);
      corrected to index w by within-row column index. STANDALONE RMSNorm HIP
      kernel on ROCm. NOT host.

    * grim_add_rms_norm (lines 210-230): fused add + RMSNorm HIP kernel. "y
      = x + residual, then RMSNorm on y. Writes y_out and norm_out." Two ops
      in one launch. FUSED norm HIP kernel on ROCm.

    * grim_rmsnorm_matmul (lines 258-280): fused RMSNorm + matmul HIP kernel.
      RMSNorm on x (row-wise), then matmul with weight_mat. FUSED HIP kernel.

    * grim_mla_q_kv_norm_split (lines 358-390): MLA-style Q/KV norm split.
      "RMSNorm on q_nope and kv_nope, rope dims copied verbatim." Standalone
      HIP kernel for MLA norm split.

    * grim_silu_mul (lines 151-157): standalone SiLU*up activation kernel.

    * grim_silu_mul_backward (lines 159-172): SiLU backward kernel.

    * grim_softmax (lines 232-247): standalone softmax kernel.

    * grim_embedding (lines 249-256): standalone embedding lookup kernel.

    * grim_broadcast_bias (lines 125-132): standalone broadcast bias kernel.

    * grim_scale_bias_epilogue (lines 134-148): scale + bias epilogue kernel.

    * grim_all_reduce_accum (lines 177-190): on-device all_reduce accumulator.

    * grim_short_conv1d_causal_step (lines 298-326): short conv1d causal step.

    * grim_kda_gated_delta_rule_step (lines 328-356): KDA gated delta rule.

    * grim_split_k_reduction (lines 282-296): split-K reduction kernel.

  Also, source_asm.rs (full file read) has a compute_kernel_source() function
  (lines 3-31) that ASSEMBLES the full kernel source by concatenating all the
  individual KERNEL_SOURCE literals from each module. This includes:
    - compute_kernels::OTHER_KERNEL_SOURCE (which contains grim_rope,
      grim_rope_yarn, grim_rms_norm, grim_add_rms_norm, and all the others)
    - qkv_attention::KERNEL_SOURCE
    - charon::KERNEL_SOURCE
    - wmma_gemm::KERNEL_SOURCE
    - and all the other kernel modules.
  And source_asm.rs has tests (lines 34-85) that ASSERT these kernels are
  present: "assert!(src.contains(\"grim_rms_norm\"))" (line 42),
  "assert!(src.contains(\"grim_qkv_attention\"))" (line 44),
  "assert!(src.contains(\"grim_wmma_gemm\"))" (line 46),
  "assert!(src.contains(\"grim_dequant_q8_0\"))" (line 48), and the
  kernel_source_has_no_duplicate_device_fn_definitions test (lines 67-84).

  So grim DEFINITELY has standalone norm HIP kernels (rms_norm, add_rms_norm
  fused, rmsnorm_matmul, mla_q_kv_norm_split) and standalone rope HIP kernels
  (rope, rope_yarn) on ROCm. The doc's claim "no norm or rope HIP kernel at
  all" is FALSE.

  What the doc got RIGHT (implicit): grim does NOT have the TRIPLE-FUSED
  QK-norm + RoPE + KV-insert kernel (fused_qknorm_rope_kernel.cu model).
  grim has the individual pieces and 2-op fusions, but not the triple fusion
  in one launch. The gap is FUSION DEPTH, not absence.

  What the doc WRONGLY claimed:
    - "norm on host/fused" → FALSE. Norm is a real HIP kernel (rms_norm,
      add_rms_norm fused, rmsnorm_matmul, mla_q_kv_norm_split).
    - "RoPE on host/fused" → FALSE. RoPE is a real HIP kernel (rope,
      rope_yarn).
    - "silu fused into charon" → PARTIALLY TRUE but misleading. grim_silu_mul
      is a standalone SiLU kernel in compute_kernels.rs (not just fused into
      charon). The doc implied SiLU only exists inside charon, but there's a
      standalone grim_silu_mul too.
    - "Vulkan HAS rms_norm.comp, rope.comp, silu_mul.comp -- but those are
      Vulkan, not ROCm" → TRUE that Vulkan has them, but FALSE that ROCm
      doesn't (ROCm has them in compute_kernels.rs).

VERDICT:
  - "NO standalone norm/rope HIP kernels on ROCm" → FALSE. Grim has them.
    This section is not just stale — it's factually wrong. It needs complete
    rewriting.
  - CORRECTED GAP: "Grim HAS standalone RMSNorm (grim_rms_norm), fused
    add+rms_norm (grim_add_rms_norm), fused rmsnorm+matmul (grim_rmsnorm_
    matmul), plain RoPE (grim_rope), partial/YaRN RoPE (grim_rope_yarn), MLA
    norm split (grim_mla_q_kv_norm_split), standalone SiLU (grim_silu_mul),
    standalone softmax (grim_softmax) — all real HIP kernels in
    compute_kernels.rs. The gap is FUSION DEPTH: grim lacks the triple-fused
    QK-norm + RoPE + KV-insert kernel that vLLM's fused_qknorm_rope_kernel.cu
    implements (one launch doing norm(Q), norm(K), RoPE both, KV-insert)."
  - Section 3 should be rewritten to reflect that norm/rope kernels EXIST and
    the port target is the TRIPLE FUSION, not the components.

--------------------------------------------------------------------------------
SECTION 4 — Skinny GEMM (LLMM1 / wvSplitK / wvSplitKrc / wvSplitKQ)
--------------------------------------------------------------------------------

DOC CLAIM:
  "NO skinny GEMM path."
  "grim's GEMM is dense tiled/WMMA/fused-dequant."

VERIFIED STATE:

  Content search "skinny|LLMM1|wvSplit|wvmSplit" in crates/grim-backend-
  rocm/src/kernels/ → ZERO matches. No file with "skinny", "LLMM1",
  "wvSplit", or "wvmSplit" in its name or content in the kernels dir.

  Content search "matrix.vector|vector.product|1.*N|batch.*1|decode.*Gemm|
  M.*1|thread.*N.*=.*1|block.*N.*=.*1|single.*column" in kernels/:
    - No "matrix-vector" or "vector-product" terms found.
    - Some matches for "batch.*1" etc. but these are about batch=1 in
      attention/conv contexts (e.g. tree_attention batch_idx, conv1d batch),
      NOT about skinny/matvec GEMM dispatch.
    - wmma_gemm.rs does tile_row * 16 >= M / tile_col * 16 >= N — this is
      the standard WMMA tile grid, NOT a small-M dispatch (it handles any M,
      small or large, via the tile grid).
    - fp8_gemm_rdna4.rs is "tiled 16x16 GEMM, float inputs" for gfx1200/gfx1100
      — dense tiled, not skinny.

  So the claim "NO skinny GEMM path" is NOT REFUTED by this search. There's
  no skinny GEMM named file and no matrix-vector product pattern found.

  BUT: the search did NOT positively confirm the claim either. It checked for
  PRESENCE of the term "skinny/LLMM1/wvSplit" (absence found) and for PRESENCE
  of matrix-vector patterns (absence found). It did NOT check whether the
  EXISTING dense GEMM kernels have a small-M or matrix-vector dispatch path
  INSIDE them (e.g. a branch "if (M == 1) { do matvec; } else { do dense;
  }"). Such a path would not contain the term "skinny" or "LLMM1" but would
  still be a skinny/matrix-vector capability.

  wmma_gemm.rs is a pure WMMA tile GEMM (16x16 tiles, 2D grid) — no
  matrix-vector special case visible. fp8_gemm_rdna4.rs is pure tiled 16x16.
  fused_dequant_gemm.rs is per-element (M*N threads) — this is effectively a
  dense GEMM where each thread does one output element, which for M=1,N=K
  would be a matvec but there's no SPECIAL treatment (no CuCount, no wave-
  count tuning, no wvSplitK-style optimization). So the existing GEMMs do NOT
  have the optimized skinny/matrix-vector path that vLLM's wvSplitK provides.

  Tentative conclusion: the claim "NO skinny GEMM path" is PROBABLY TRUE
  (no named skinny kernel, no matrix-vector pattern, existing GEMMs are dense
  without small-M optimization), but it has NOT been positively confirmed. A
  dedicated check would be: search the existing GEMM kernels for any "if (M
  == 1)" or "if (M < threshold)" branch that dispatches to a different
  algorithm, and confirm there is none. The absence of "skinny/LLMM1/wvSplit"
  terms plus the absence of matrix-vector patterns in the search is suggestive
  but not definitive proof.

VERDICT:
  - "NO skinny GEMM path": UNTRIED / TENTATIVELY TRUE. Not refuted by this
    pass (no matches for skinny/LLMM1/wvSplit, no matrix-vector patterns
    found). The existing GEMMs are dense (WMMA tiled, RDNA4 tiled, fused
    dequant per-element) without small-M optimization. BUT: not positively
    confirmed — a dedicated check (confirming no "if (M == 1)" branch in the
    GEMM kernels) would be needed to be certain.
  - The doc should mark this as "not confirmed this pass — worth a dedicated
    positive check" rather than stating it as fact.

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------

Section 1 (PagedAttention/MFMA):
  - "STUB...incomplete" for paged: FALSE (paged kernel is complete).
  - "No MFMA": TRUE (no MFMA in attention), but understates that wmma_gemm.rs
    has a real gated MFMA scaffold with placeholder.
  - "No FP8 KV": TRUE.
  - Correction needed: paged is real, MFMA scaffold exists elsewhere.

Section 2 (W4A16 GPTQ GEMM):
  - "NO W4A16 GPTQ at all" for inference: TRUE (no inference dequant+GEMM).
  - gptq_kernel.rs EXISTS at src/gptq_kernel.rs (NOT in kernels/) and
    implements OFFLINE calibration (correction + scale fit), NOT inference
    dequant+GEMM.
  - NAMING COLLISION CONFIRMED: searching "gptq" in crate will surface
    gptq_kernel.rs and may mislead. Doc should warn.
  - Correction: be precise about "inference-time" vs "offline calibration",
    and note the naming collision.

Section 3 (Fused norm/RoPE/KV-insert):
  - "NO standalone norm/rope HIP kernels on ROCm": FALSE. Grim HAS them.
  - "norm on host/fused, RoPE on host/fused": FALSE. Both are real HIP
    kernels in compute_kernels.rs.
  - This section is factually WRONG and needs complete rewriting.
  - CORRECTED GAP: triple-fused QK-norm+RoPE+KV-insert (fusion depth), not
    absence of norm/rope kernels.

Section 4 (Skinny GEMM):
  - "NO skinny GEMM path": UNTRIED / TENTATIVELY TRUE. Not refuted by this
    pass (no matches for skinny/LLMM1/wvSplit, no matrix-vector patterns).
    Existing GEMMs are dense without small-M optimization. BUT not positively
    confirmed — needs dedicated check for "if (M == 1)" branches.
  - Doc should mark as unconfirmed.

--------------------------------------------------------------------------------
FILES READ FOR THIS VERDICT
--------------------------------------------------------------------------------

  - crates/grim-backend-rocm/src/kernels/qkv_attention.rs (full, 726 lines)
  - crates/grim-backend-rocm/src/kernels/compute_kernels.rs (full, 437 lines)
  - crates/grim-backend-rocm/src/kernels/source_asm.rs (full, 85 lines)
  - crates/grim-backend-rocm/src/kernels/wmma_gemm.rs (full, 408 lines)
  - crates/grim-backend-rocm/src/gptq_kernel.rs (full, 286 lines)
  - crates/grim-backend-rocm/src/lib.rs (gptq_kernel mod + re-export)
  - crates/grim-backend-rocm/src/device/cubecl.rs (gptq_correction fn)
  - crates/grim-backend-rocm/tests/device_cubecl.rs (gptq_correction test,
    lines 327-353)
  - Content searches:
      "gptq" across crate → 24 matches (gptq_kernel.rs + lib.rs + device_cubecl
      + tests, NO matches in kernels/)
      "skinny|LLMM1|wvSplit|wvmSplit" in kernels/ → 0 matches
      "matrix.vector|vector.product|1.*N|batch.*1|decode.*Gemm|M.*1|thread.*N.*=
      .*1|block.*N.*=.*1|single.*column|vector.*product" in kernels/ → 0 relevant
      matches (batch=1 etc. are attention/conv context, not matvec GEMM)
      "w4a16|half2(1024|q_gemm_rdna3|gptq_gemm" across crate → 0 matches

--------------------------------------------------------------------------------
VERDICT COMPLETE
--------------------------------------------------------------------------------
