GRIM-TO-VLLM PORT DOC: SECTIONS 1–4 VERDICT
==============================================

This verdict checks the doc's claims about grim's current state against
the actual source files in crates/grim-backend-rocm/src/kernels/.
Files read for this verdict: qkv_attention.rs (full), compute_kernels.rs
(full), source_asm.rs (full), gptq_kernel.rs (search — file not found in
kernels dir), plus content searches for norm/rope/skinny symbols.


SECTION 1 — PagedAttention / QK MFMA attention
------------------------------------------------

DOC CLAIM:
  "STUB paged path (BlockTableEntry, page walk) -- exists in source but
   is incomplete vs vLLM's real paged attention."
  "No MFMA. No FP8 KV."
  "grim_qkv_attention_paged exists but is incomplete."

VERIFIED STATE (qkv_attention.rs):

  - grim_qkv_attention_paged IS a real, complete kernel, not a stub.
    Lines 208–357 in qkv_attention.rs contain a full HIP kernel with:
      * BlockTableEntry struct (block_id, page_size) — line 203–206
      * Full page-walk math: b = j / page_size, t = j % page_size,
        physical_token_idx = entry.block_id * page_size + t — lines 281–284
      * Causal guard: if (j > abs_i || j >= kv_seq_len) break; — line 278
      * Online softmax with running_max/running_sum, wave merge in LDS
        s_max/s_sum/s_acc[8][256], same pattern as the non-paged kernel
      * GQA mapping: q_per_kv, kv_head — lines 227–228
      * Q layout [batch, num_heads, head_dim], K/V pages as
        [num_pages, page_size, num_kv_heads, head_dim] — lines 231, 286

    This is a real paged attention kernel. The doc's "STUB...incomplete"
    claim is WRONG. The page-walk math is real and complete.

  - MFMA: the KERNEL_SOURCE in qkv_attention.rs (lines 1–724) does NOT
    contain any __builtin_amdgcn_mfma* instructions. The file ends at
    line 724 after tree_attention's wave-0 merge. No MFMA anywhere in
    this file.

    HOWEVER, the broader grim-wmma crate (unread this pass, but
    previously confirmed) has a real gated MFMA scaffold:
      * wmma_gemm.rs has `#if defined(__gfx1200__)` / `#if defined(__gfx1201__)`
        gates with a placeholder instruction that is honestly labeled as a
        scalar fallback (line 332–336 in the earlier read: "On real CDNA
        hardware, these 32 FP8→F32 conversions happen via the mfma
        instruction itself. This scalar fallback ensures compilation on
        non-gfx1200 targets").
      * So MFMA is not "no MFMA at all" — there's a real gated scaffold
        with a placeholder instruction, honestly labeled.

  - FP8 KV: not in qkv_attention.rs. True. No k_scale/v_scale FP8 KV
    path in this file.

VERDICT FOR SECTION 1:
  - "STUB...incomplete" for paged attention: FALSE. The paged kernel is
    real and complete. Correct the doc to: "grim_qkv_attention_paged is a
    real paged-attention kernel with full page-walk math; gap vs vLLM is
    breadth/features (vLLM's is multi-template with MFMA + FP8 KV + ALIBI
    + GQA ratio variants), not presence."
  - "No MFMA": PARTIALLY TRUE but understated. There's no MFMA
    INSTRUCTION in the attention path, but there IS a real gated MFMA
    scaffold in wmma_gemm.rs with an honestly-labeled placeholder. The gap
    is narrower than "no MFMA at all." Correct to: "No MFMA instruction in
    the attention kernel yet; a real gated MFMA scaffold exists elsewhere
    in grim-wmma (wmma_gemm.rs) with a placeholder that compiles on
    non-gfx1200 targets."
  - "No FP8 KV": TRUE. No FP8 KV path in qkv_attention.rs.


SECTION 2 — W4A16 GPTQ GEMM (dense + WMMA, RDNA3)
----------------------------------------------------

DOC CLAIM:
  "NO W4A16 GPTQ at all."
  "grim's quantization universe is K-quant/IQ-quant + FP8/MXFP4/MXFP8."

VERIFIED STATE:

  - Search for gptq_kernel.rs in crates/grim-backend-rocm/src/kernels/
    returned NO matches — file not found in the kernels directory. (It may
    live elsewhere in the crate, or not exist in this checkout.)

  - Content search for "w4a16|gptq.*gemm|w4a16.*dequant|half2(1024"
    in the kernels dir returned NO matches (no w4a16 dequant bit-trick,
    no GPTQ GEMM entry).

  - So for the INFERENCE-TIME w4a16 dequant+GEMM that section 2 is about,
    the claim "NO W4A16 GPTQ at all" is essentially TRUE — there's no
    w4a16 dequant kernel and no w4a16 GEMM kernel in the kernels dir.

  - NAMING COLLISION RISK: if gptq_kernel.rs exists anywhere in the crate
    (just not in the kernels/ subdirectory), it likely implements a
    DIFFERENT thing — offline weight-correction (GPTQ's Hessian-diagonal
    update algorithm) and scale search, NOT inference-time dequant+GEMM.
    This is a real risk: anyone checking "is GPTQ done" by searching for
    the term "gptq" in the crate will find gptq_kernel.rs and could
    wrongly conclude section 2 is complete or in-progress on the right
    thing.

VERDICT FOR SECTION 2:
  - "NO W4A16 GPTQ at all" for inference GEMM: ESSENTIALLY TRUE. No
    w4a16 dequant+GEMM kernel in the kernels dir. The doc's claim is
    correct for what section 2 is actually about (inference-time
    dequant+GEMM).
  - BUT the doc should add a note: "If a file named gptq_kernel.rs exists
    in the crate, it likely implements offline GPTQ calibration (Hessian-
    diagonal weight correction + scale search), which is a DIFFERENT thing
    from the inference-time w4a16 dequant+GEMM in section 2. The naming
    overlap risks confusion: searching 'gptq' in the crate may surface the
    calibration file and wrongly suggest section 2 is in progress or
    complete. Clarify the distinction when porting section 2."


SECTION 3 — Fused QK-norm + RoPE + KV-insert kernels
-------------------------------------------------------

DOC CLAIM:
  "NO standalone norm/rope/activation HIP kernels on ROCm."
  "grim's ROCm side has no norm or rope HIP kernel."
  "norm on host/fused, RoPE on host/fused, silu fused into charon."

VERIFIED STATE (compute_kernels.rs, full file read):

  - grim_rope: YES, real HIP kernel. Lines 35–62. Plain full-rotary RoPE.
    Contract: 3-D input [B, S, D], positions[si] per step, rotation to
    pairs (x[i], x[half+i]). One thread per (batch, step, dim-half-pair)
    element. This is a standalone RoPE HIP kernel on ROCm.

  - grim_rope_yarn: YES, real HIP kernel. Lines 79–123. Partial-rotary +
    YaRN kernel. Handles rotary_dim <= d (partial) and pre-computed
    YaRN-ramp frequencies (inv_freq param, mscale param). Two-pass: rotate
    [0, rotary_half) pairs, then copy non-rotary dims [2*rotary_half, d)
    verbatim. This is a standalone partial-rotary/YaRN HIP kernel on ROCm.

  - grim_rms_norm: YES, real HIP kernel. Lines 192–208. RMSNorm: compute
    variance = sum(x*x)/row_len, rms = sqrt(variance + eps), out =
    x * w[col] / rms. Fixed a bug where w was indexed by global linear
    index instead of within-row column index (lines 203–207 comment).

  - grim_add_rms_norm: YES, real HIP kernel. Lines 210–230. Fused
    add + RMSNorm: y = x + residual, then RMSNorm on y. Writes y_out and
    norm_out. This is a fused norm kernel on ROCm.

  - grim_silu_mul: YES, real HIP kernel. Lines 151–157. silu(g) * up.
    Standalone activation kernel.

  - grim_silu_mul_backward: YES, real HIP kernel. Lines 159–172. SiLU
    backward with d_silu = sigmoid(e) * (1 + e*(1-sigmoid(e))).

  - grim_softmax: YES, real HIP kernel. Lines 232–247. Standalone softmax.

  - grim_embedding: YES, real HIP kernel. Lines 249–256. Standalone
    embedding lookup.

  - grim_rmsnorm_matmul: YES, real HIP kernel. Lines 258–280. Fused
    RMSNorm + matmul.

  - grim_mla_q_kv_norm_split: YES, real HIP kernel. Lines 358–390. MLA-
    style Q/KV norm split (RMSNorm on q_nope and kv_nope, rope dims
    copied verbatim). This is an MLA-adjacent norm kernel.

  - grim_split_k_reduction, grim_short_conv1d_causal_step,
    grim_kda_gated_delta_rule_step, grim_broadcast_bias,
    grim_scale_bias_epilogue, grim_all_reduce_accum: all real HIP kernels.

  So grim HAS standalone norm HIP kernels (rms_norm, add_rms_norm fused,
  rmsnorm_matmul, mla_q_kv_norm_split) and standalone rope HIP kernels
  (rope, rope_yarn) on ROCm. The doc's claim "no norm or rope HIP kernel
  at all" is FALSE.

  What grim does NOT have (that vLLM has): the TRIPLE-FUSED
  QK-norm + RoPE + KV-insert kernel (fused_qknorm_rope_kernel.cu) where
  norm(Q), norm(K), RoPE both, and KV-insert all happen in ONE kernel
  launch. grim has the individual pieces (norm, rope) as separate kernels;
  it does not have the triple fusion. But the doc claims "no norm or rope
  kernel" which is wrong — the gap is fusion depth, not absence.

VERDICT FOR SECTION 3:
  - "NO standalone norm/rope HIP kernels on ROCm": FALSE. Grim has
    grim_rms_norm, grim_add_rms_norm (fused), grim_rope, grim_rope_yarn,
    grim_rmsnorm_matmul, grim_mla_q_kv_norm_split, grim_silu_mul, and
    grim_softmax as real HIP kernels in compute_kernels.rs.
  - "norm on host/fused, RoPE on host/fused": FALSE. Norm and RoPE are
    real HIP kernels, not host-side.
  - CORRECTED CLAIM: "Grim has standalone RMSNorm (grim_rms_norm), fused
    add+rms_norm (grim_add_rms_norm), plain RoPE (grim_rope), and YaRN/
    partial-rotary RoPE (grim_rope_yarn) as real HIP kernels on ROCm.
    What grim does NOT have is the triple-fused QK-norm+RoPE+KV-insert
    kernel that vLLM's fused_qknorm_rope_kernel.cu implements. The gap is
    fusion depth (one-launch triple fusion), not absence of norm/rope
    kernels."
  - This section is not just stale — it's actively wrong. It needs to be
    rewritten to reflect that norm and rope kernels exist.


SECTION 4 — Skinny GEMM (LLMM1 / wvSplitK / wvSplitKrc / wvSplitKQ)
----------------------------------------------------------------------

DOC CLAIM:
  "NO skinny GEMM path."
  "grim's GEMM is dense tiled/WMMA/fused-dequant."

VERIFIED STATE:

  - Content search for "skinny|LLMM1|wvSplit" in
    crates/grim-backend-rocm/src/kernels/ returned ZERO matches.

  - No skinny-GEMM-named file surfaced in the diff or in the gptq|skinny
    grep.

  - So the claim "NO skinny GEMM path" is NOT REFUTED by this check. It
    stands as unrefuted, but it has NOT been positively confirmed either
    (I checked for absence, not presence of a true skinny GEMM).

  - The doc's implicit claim that grim's GEMM is "dense tiled/WMMA/
    fused-dequant" is consistent with what's in the kernels dir (wmma_gemm,
    fp8_gemm_rdna4, fused_dequant_gemm, q4k/q5k/q6k/q2k/q3k/iq_gemm —
    all dense or fused-dequant, no skinny/matrix-vector named kernels).

VERDICT FOR SECTION 4:
  - "NO skinny GEMM path": UNTRIED. Not refuted by this pass (no
    skinny/LLMM1/wvSplit symbols found), but not positively confirmed
    either. The doc should either:
      (a) Run a dedicated positive check (search for matrix-vector GEMM
          patterns, for M=1 or small-M dispatch paths in the existing
          GEMM kernels) before trusting this claim, OR
      (b) Mark it as "not confirmed this pass — worth a dedicated search."
  - Tentatively still accurate based on absence of evidence, but flagged
    as unconfirmed.


SUMMARY OF REQUIRED DOC CORRECTIONS
=====================================

1. SECTION 1: 
   - "STUB...incomplete" → WRONG. Paged kernel is real and complete.
     Rewrite to: "grim_qkv_attention_paged is a real paged-attention kernel
     with full page-walk math; the gap vs vLLM is feature breadth (vLLM's
     is multi-template with MFMA + FP8 KV + ALIBI + GQA ratio variants),
     not presence."
   - "No MFMA" → UNDERTATED. There's no MFMA instruction in the attention
     path, but a real gated MFMA scaffold exists in wmma_gemm.rs with an
     honestly-labeled placeholder. Correct to: "No MFMA instruction in the
     attention kernel yet; a real gated MFMA scaffold exists in
     wmma_gemm.rs with a placeholder that compiles on non-gfx1200 targets."

2. SECTION 2:
   - "NO W4A16 GPTQ at all" → ESSENTIALLY TRUE for inference GEMM.
   - ADD a naming-collision note: if gptq_kernel.rs exists in the crate,
     it likely implements offline GPTQ calibration (Hessian-diagonal
     weight correction + scale search), NOT inference-time dequant+GEMM.
     Clarify the distinction to avoid confusion when searching "gptq" in
     the crate.

3. SECTION 3:
   - "NO standalone norm/rope HIP kernels on ROCm" → FALSE. Grim has
     grim_rms_norm, grim_add_rms_norm, grim_rope, grim_rope_yarn,
     grim_rmsnorm_matmul, grim_mla_q_kv_norm_split, grim_silu_mul,
     grim_softmax as real HIP kernels in compute_kernels.rs.
   - REWRITE the section entirely. The gap is fusion depth (triple-fused
     QK-norm+RoPE+KV-insert), not absence of norm/rope kernels.

4. SECTION 4:
   - "NO skinny GEMM path" → UNTRIED. Not refuted, not confirmed. Mark as
     "not confirmed this pass — worth a dedicated search" or run a
     dedicated positive check before trusting.


FILES READ FOR THIS VERDICT
=============================

  - crates/grim-backend-rocm/src/kernels/qkv_attention.rs (full, 724 lines)
  - crates/grim-backend-rocm/src/kernels/compute_kernels.rs (full, 437 lines)
  - crates/grim-backend-rocm/src/kernels/source_asm.rs (full, 85 lines)
  - Content search: grim_rms_norm|grim_add_rms_norm|grim_rope|grim_rope_yarn
    → 12 matches, all in compute_kernels.rs (real kernels confirmed)
  - Content search: skinn|LLMM1|wvSplit → 0 matches (no skinny GEMM found)
  - Search for gptq_kernel.rs in kernels dir → file not found


VERDICT COMPLETE.
