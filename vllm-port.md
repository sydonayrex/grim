# vllm-to-grim: what to port from vLLM's AMD ROCm kernel surface


Purpose: take the kernel comparison (voogle.md) and turn it into porting decisions.
For each vLLM kernel type: port / don't-port / maybe, why, what to port (approach
vs code), priority, and the grim gap it closes. UPDATED: grim has decided to
support GPTQ-w4a16 models on RDNA3, so section 2 is flipped from NO to PORT P0.

Audience: grim ROCm backend team. Target arch: gfx1036 (RDNA2, W64) primary;
W32 iGPU secondary; gfx1100 (RDNA3) GPTQ target; gfx942/gfx950 (CDNA) secondary.
vLLM's AMD surface is C++/hipcc; grim is Rust+HIP. Where a vLLM kernel is worth
porting, we port the ALGORITHM/APPROACH as a new `pub const KERNEL_SOURCE: &str =
r#"..."#` HIP literal in grim, not the C++ file. Where the vLLM kernel is C++ only
and not worth it, we skip it.

--------------------------------------------------------------------------------
0. HOW TO READ THIS DOC
--------------------------------------------------------------------------------

Each section is one vLLM kernel area. The header line says the decision:

  [PORT  P0] ...   -> port now, top priority
  [PORT  P1] ...   -> port soon, high value
  [PORT  P2] ...   -> port later, nice-to-have
  [NO    --] ...   -> don't port; reason given
  [MAYBE ??] ...   -> needs a decision; conditions listed

"Approach" means: study the vLLM kernel, understand the algorithm, write a grim
HIP kernel that implements the same thing. "Code" would mean copying the C++; we
don't do that. grim keeps its Rust+HIP-literal style.

Every PORT section now includes a MUTATION-RESISTANT TDD subsection (see section
18 for the general framework; each port restates the four test classes with the
specific values for that kernel). NO sections still include a one-line reason.

The cross-reference column points to the vLLM source file(s) that are the source
of truth for that kernel. Read those before writing the grim port.

--------------------------------------------------------------------------------
1. PagedAttention / QK MFMA attention  [PORT  P0]
--------------------------------------------------------------------------------

vLLM source: csrc/rocm/attention.cu (3717 lines)
vLLM Python wrapper: vllm._custom_ops.paged_attention_rocm, vllm._rocm_C
vLLM backend selector: vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend

What vLLM has:
- Multi-template paged_attention kernel. Templated on scalar_t, cache_t,
  KV_DTYPE (auto vs fp8), OUTT, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS,
  ALIBI_ENABLED, GQA_RATIO, MFMAType.
- Two kernel families: LL4MI (256-thread, warp-level, shared_logits
  [warp][4][16][4], two-pass exp normalization) AND gfx11 wave32
  mfma16x16x16 variant.
- MFMA for QK on f16/bf16 + optional FP8 KV cache MFMA (gfx942/gfx950,
  __HIP__FP8MFMA__, FP8_E4M3_SCALE_TARGET=224, q_max warpReduceMax).
- PagedAttention: block_tables, seq_lens, query_start_loc.
- Rich v1 backend selector: ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN,
  TRITON_ATTN, TURBOQUANT, FLASH_ATTN, TORCH_SDPA + MLA variants.

What grim has:
- qkv_attention.rs: Phase-1 causal GQA, online softmax, wave-aware (W64 vs W32),
  LDS s_max/s_sum/s_acc[8][256], grid=(seq_len, num_heads,1), block=(256,1,1),
  head-dim cap 256.
- grim_qkv_attention_paged: REAL, COMPLETE paged-attention kernel (NOT a stub).
  Has full page-walk math: BlockTableEntry (block_id, page_size), b = j / page_size,
  t = j % page_size, physical_token_idx = entry.block_id * page_size + t, K/V pages
  as [num_pages, page_size, num_kv_heads, head_dim], causal guard (j > abs_i ||
  j >= kv_seq_len) break, online softmax with same wave-merge LDS pattern as the
  non-paged kernel. Verified present and complete in qkv_attention.rs lines 203-357.
  The gap vs vLLM is FEATURE BREADTH (vLLM's is multi-template with MFMA + FP8 KV +
  ALIBI + GQA ratio variants), not presence/absence of paging.
- grim_tree_attention: tree attention variant.
- MFMA: no MFMA instruction in the attention kernel path yet. BUT a real gated MFMA
  scaffold exists elsewhere in grim-wmma (wmma_gemm.rs) with an honestly-labeled
  placeholder instruction that compiles on non-gfx1200 targets. So "no MFMA at all"
  understates the gap — there's a scaffold waiting for a real CDNA MFMA instruction
  to be slotted in. For gfx1100+ (RDNA3) secondary arch, MFMA for QK is worth porting
  from vLLM as an optional gated path.
- FP8 KV: no FP8 KV path (k_scale/v_scale, FP8_E4M3_SCALE_TARGET) in qkv_attention.rs.
  True gap. Worth porting for CDNA (gfx942/gfx950) secondary arch.
- No real paging semantics matching vLLM's — CORRECTED: grim HAS real paging semantics
  (the paged kernel is complete); what it lacks is vLLM's multi-template breadth.

GAP: grim HAS a real paged attention kernel. The gap is breadth: vLLM's attention.cu
is a multi-template PagedAttention with MFMA + FP8 KV + ALIBI + GQA ratio variants
AND a separate gfx11 wave32 mfma16x16x16 family, plus a rich backend selector. grim's
paged kernel is one fixed-configuration kernel. The work is to EXTEND grim's paged
kernel (add MFMA QK gated on gfx1100+, add FP8 KV for CDNA, possibly add ALIBI) rather
than to build paging from scratch.

PORT DECISION: PORT the approach for the EXTENSIONS (MFMA QK for gfx1100+, FP8 KV for
CDNA), not the base paging (which already exists). Don't copy the C++.

Specifically, port:
(a) MFMA for QK. vLLM uses __builtin_amdgcn_mfma_f32_4x4x4f16 (f16) and
    __builtin_amdgcn_mfma_f32_16x16x16f16 on gfx11 (RDNA3). grim targets gfx1036
    (RDNA2) which does NOT have MFMA. For grim's primary arch, MFMA is N/A -- grim
    stays with the scalar dot-product path. For gfx1100+ (RDNA3) secondary arch, port
    the MFMA QK path from vLLM as an optional gated path. Note: grim already has a
    gated MFMA scaffold in wmma_gemm.rs (placeholder instruction, compiles on non-
    gfx1200), so the scaffolding is real -- what's needed is the real MFMA instruction
    for the attention path on gfx1100+.
(b) FP8 KV cache attention. vLLM's attention.cu supports KV_DTYPE==fp8 with
    k_scale/v_scale, FP8_E4M3_SCALE_TARGET normalization, per-warp q_max reduction.
    grim's qkv_attention.rs is f16/bf16 only. For CDNA targets (gfx942/gfx950) this
    is worth porting; grim's primary gfx1036 doesn't have FP8 MFMA. Mark as a
    secondary-arch enhancement.
(c) Multi-backend attention selector. vLLM's v1 has a rich priority list. grim
    doesn't need a 8-way menu yet; it has one real paged kernel. Port the paged
    kernel extensions first; the menu can come later if grim wants flash/turboquant/
    MLA backends. Don't over-build the selector before the extensions exist.

What NOT to port:
- The C++ file itself. grim writes HIP literals.
- The full 8-way backend menu. Port the paged kernel extensions; the menu is
  scope creep for round 1.

Priority: P0. Grim's attention HAS paging; the gap is extending it with MFMA + FP8 KV.

RISK: vLLM's attention.cu is 3717 lines of C++ templates. Don't read it all at
once. Read the LL4MI kernel path first (the 256-thread warp-level one), then the
gfx11 mfma16x16x16 variant, then the FP8 KV path. Port the MFMA QK extension and
FP8 KV extension as gated additions to grim's existing paged kernel.

Skills: rust-ffi (HIP FFI for the new kernel launch: hipModuleLoad/hipModuleGetFunction
  /hipModuleLaunchKernel, rocblas if used, status checking, the dlopen-vs-link-time
  decision for which ROCm runtime to load), rust-gpu (GPU kernel correctness gates
  before performance: LDS budget, wave size, head-dim cap, the parity-vs-reference
  discipline), rocm-kernels (AMD GPU kernel tuning: RDNA2 wave64 constraints, LDS
  sizing, the wave-aware partitioning pattern vLLM uses and grim should replicate).

What NOT to port:
- The C++ file itself. grim writes HIP literals.
- The full 8-way backend menu. Port the paged attention kernel; the menu is
  scope creep for round 1.

Priority: P0. This is the single biggest gap between grim and vLLM. grim's
attention is Phase-1; vLLM's is production PagedAttention.

RISK: vLLM's attention.cu is 3717 lines of C++ templates. Don't read it all at
once. Read the LL4MI kernel path first (the 256-thread warp-level one), then the
gfx11 mfma16x16x16 variant, then the FP8 KV path. Port the LL4MI approach as the
primary grim path (matches gfx1036 wave64), MFMA as a gated gfx1100+ addition.

Skills: rust-ffi (HIP FFI for the new kernel launch: hipModuleLoad/hipModuleGetFunction
  /hipModuleLaunchKernel, rocblas if used, status checking, the dlopen-vs-link-time
  decision for which ROCm runtime to load), rust-gpu (GPU kernel correctness gates
  before performance: LDS budget, wave size, head-dim cap, the parity-vs-reference
  discipline), rocm-kernels (AMD GPU kernel tuning: RDNA2 wave64 constraints, LDS
  sizing, the wave-aware partitioning pattern vLLM uses and grim should replicate).

Mutation-resistant TDD (see section 18 for the full framework):
  - RED: source-content test asserts the paged kernel entry symbol exists in
    KERNEL_SOURCE. If the kernel is omitted, this fails RED. GREEN: add the literal.
  - RED: source-string test asserts the paged attention math pattern (block_table
    walk, page_size decomposition, physical_token_idx formula). If someone ports a
    non-paged attention kernel and labels it "paged", this fails RED. GREEN: the
    literal contains the real paging math.
  - RED: CPU parity oracle test. Reference: vLLM's paged_attention result (computed
    on CPU via the same block_table+paging+online-softmax algorithm) or a pure-Rust
    re-implementation. The HIP kernel output is read back with hipMemcpyDtoH after
    hipDeviceSynchronize. If the kernel produces wrong paged attention (e.g. wrong
    page walk, wrong softmax), the oracle catches it. If the oracle is too loose
    (e.g. only checks max_abs < 1e-3 against a wrong reference), mutation replaces
    the algorithm with something that still passes -- so the oracle must be the
    correct algorithm, not an approximate one.
  - RED: metamorphic test. vLLM's paged attention on a known input (e.g. a 2-batch,
    2-page, GQA=2 case) produces output X. The grim paged kernel on the same input
    must produce output Y within tolerance of X. If someone swaps the kernel for a
    non-paged kernel, Y diverges from X on the paging dimension. The test must use
    an input where paging matters (seq_lens that span page boundaries), otherwise
    a non-paged kernel also passes.

--------------------------------------------------------------------------------
2. W4A16 GPTQ GEMM (dense + WMMA, RDNA3)  [PORT  P0]  *** UPDATED ***
--------------------------------------------------------------------------------

vLLM source: csrc/rocm/q_gemm_rdna3.cu (780 lines), csrc/rocm/q_gemm_rdna3_wmma.cu
(2165 lines), csrc/rocm/moe_q_gemm_rdna3.cu (639 lines), csrc/rocm/qdq_4_rdna3.cuh
(239 lines)

What vLLM has:
- W4A16 GPTQ dense GEMM for RDNA3 (gfx1100): scalar path (M<16) + WMMA path
  (M>=16 via gptq_gemm_rdna3 op selecting by size_m).
- WMMA variant: gemm_q4_wmma_kernel_16x16_1w (v1, 16M x 16N, 1 wave) +
  gemm_q4_wmma_kernel_32x16_2w (v2, 32M x 16N, 2 waves, double-buffered LDS).
- K-split heuristic (compute_wmma_k_split, compute_wmma_k_split_mn).
- Fused W4A16 MoE: moe_q_gemm_rdna3.cu.
- Precise dequant variant in WMMA TU (fp16 sub-first, bf16->bf16 via fp32 internal).

What grim has:
- NO W4A16 GPTQ at all for inference-time dequant+GEMM. grim's quantization universe
  was K-quant/IQ-quant + FP8/MXFP4/MXFP8. UPDATED: grim has decided to support
  GPTQ-w4a16 models on RDNA3, so this section is now a required port.
- NAMING COLLISION WARNING: if a file named gptq_kernel.rs (or any file containing
  "gptq" in its name or content) exists anywhere in the grim-backend-rocm crate, it
  likely implements a DIFFERENT thing -- offline GPTQ calibration (Hessian-diagonal
  weight correction + scale search), NOT inference-time w4a16 dequant+GEMM. This is
  a real confusion risk: anyone checking "is GPTQ done" by searching for "gptq" in the
  crate may find that file and wrongly conclude section 2 is in progress or complete
  on the right thing. When porting section 2, use clear naming that distinguishes the
  inference-time dequant+GEMM path (e.g. "w4a16_inference" or "gptq_w4a16_gemm") from
  any offline calibration file. The two are different operations and should not share
  a name.

GAP: grim had no W4A16 GPTQ path. This is now a required port because grim targets
GPTQ-w4a16 models on RDNA3.

PORT DECISION: PORT the approach. This is now P0 because grim targets RDNA3 GPTQ.

Specifically, port:
(a) W4A16 dequant primitives from qdq_4_rdna3.cuh. The fp16 bit-trick
    (half2(1024+q, 1024+q*16) via 0x64006400 OR) and bf16 variant
    (per-pair right-shift before OR with 0x43004300). These are the core dequant
    math grim needs for W4A16. Port as `pub const KERNEL_SOURCE` HIP helper
    functions _and_ as pure-Rust CPU reference functions (the bit-trick is pure
    integer math, easy to mirror on CPU for the parity oracle).
(b) Scalar W4A16 GEMM from q_gemm_rdna3.cu. M<16 scalar path: BLOCK_KN_SIZE=256,
    THREADS_X=256, 4 N columns per thread, fp16 v_dot2_f32_f16 or bf16 fp32-wide
    dot, CAS-loop packed atomic-add (64-bit word for f16, 32-bit word pair CAS for
    wmma epilogue). Port as the primary grim path for RDNA3 scalar GEMM.
(c) WMMA W4A16 GEMM from q_gemm_rdna3_wmma.cu. v1 16x16 1-wave + v2 32x16 2-wave
    double-buffered LDS. K-split heuristic. Precise dequant variant. Port as the
    WMMA path for M>=16 on gfx1100+.
(d) Fused W4A16 MoE from moe_q_gemm_rdna3.cu. routing + dequant+dot in one launch,
    same scalar/WMMA split, CAS atomic epilogue, topk_weight mul + output_topk
    reduction. Port as the MoE path for GPTQ-w4a16 models.

What NOT to port: the C++ file itself, the torch_bindings.cpp registration, the
vLLM Python wrapper. Port the algorithm as a grim HIP literal + Rust launcher.

Priority: P0. grim targets RDNA3 GPTQ-w4a16 models.

Skills: rust-gpu (WMMA via rocwmma.hpp on gfx1100+, the 16x16/32x16 tile layout,
  K-split heuristic, the double-buffered LDS pattern, the CAS atomic epilogue),
  rocm-kernels (RDNA3-specific tuning: wave32 layout, the v_dot2 intrinsic, the
  CAS-loop packed atomic, the precise dequant variant), rust-ffi (HIP JIT compilation
  of the new kernel, rocblas if used, device-side fp8/bf16 helpers).

Mutation-resistant TDD:
  - RED: source-content test asserts W4A16 dequant helper + scalar GEMM entry +
    WMMA GEMM entry + fused MoE entry all exist in KERNEL_SOURCE. If any is omitted,
    RED. GREEN: all four entries present.
  - RED: source-string test asserts the W4A16 dequant bit-trick (the 0x64006400 OR,
    the half2 construction, the bf16 per-pair shift) is in the literal. If someone
    ports a generic 4-bit unpack (not the GPTQ-specific bit-trick), this fails RED.
    GREEN: the literal contains the GPTQ bit-trick.
  - RED: CPU parity oracle test. The W4A16 dequant is pure integer/float math and
    trivially mirrorable on CPU: given W4A16 codes (nibble-per-byte, the GPTQ
    layout), scales, zeros, produce the dequantized FP16/bf16 weights on CPU, then
    do the GEMM on CPU, compare to HIP kernel output. The oracle must use the SAME
    bit-trick as the kernel (not a generic unpack), otherwise a swapped dequant
    still passes. The GEMM oracle is the CPU FP32 GEMM of the dequantized weights
    vs the HIP output (read back with hipMemcpyDtoH after hipDeviceSynchronize).
  - RED: cross-path parity test. The scalar GEMM path and the WMMA GEMM path must
    produce the same output on the same input (within fp16/bf16 tolerance). If the
    WMMA path has a layout bug (wrong wave32 lane mapping, wrong tile orientation),
    the scalar path catches it. If someone swaps the WMMA path for a scalar path
    labeled "WMMA", the test still passes -- so this test must be paired with the
    source-content test that asserts the WMMA intrinsic is actually in the literal.
  - RED: K-split heuristic test. The compute_wmma_k_split / compute_wmma_k_split_mn
    logic (ported as pure Rust) must produce the same split decisions as vLLM's for
    a set of K values. If the heuristic is wrong, the wrong number of K-splits fires
    and the output diverges from the no-split oracle.

--------------------------------------------------------------------------------
3. Fused QK-norm + RoPE + KV-insert kernels  [PORT  P1]
--------------------------------------------------------------------------------

vLLM source: csrc/libtorch_stable/fused_qknorm_rope_kernel.cu,
  csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu,
  csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu

What vLLM has:
- Fused QK-norm + RoPE + KV-insert HIP kernels. These fuse norm(x) on Q and K,
  apply RoPE, then insert into KV cache -- three ops in one kernel launch.
- Variants for DeepSeek-V4 (scaling rope) and Minimax-M3.

*** VERIFIED AGAINST GRIM SOURCE (compute_kernels.rs, full file read): ***
  The doc's prior claim that "grim's ROCm side has no norm or rope HIP kernel" is
  FALSE. grim HAS standalone norm and rope HIP kernels on ROCm, all in
  compute_kernels.rs (KERNEL_SOURCE / OTHER_KERNEL_SOURCE):

    * grim_rope (lines 35-62): standalone plain full-rotary RoPE. 3-D input [B,S,D],
      positions[si] per step, rotation to pairs (x[i], x[half+i]). One thread per
      (batch, step, dim-half-pair) element. CONTRACT: rotary_dim == d (full rotary);
      use grim_rope_yarn for partial/YaRN.

    * grim_rope_yarn (lines 79-123): standalone partial-rotary + YaRN kernel. Handles
      rotary_dim <= d (partial) and pre-computed YaRN-ramp frequencies (inv_freq param,
      mscale param). Two-pass: rotate [0, rotary_half) pairs, then copy non-rotary
      dims [2*rotary_half, d) verbatim. CONTRACT: x[d, ...], positions[S], inv_freq
      pre-computed, out[d, ...], b/s/d/rotary_half/mscale params.

    * grim_rms_norm (lines 192-208): standalone RMSNorm HIP kernel. Computes
      variance = sum(x*x)/row_len, rms = sqrt(variance + eps), out = x * w[col] / rms.
      FIXED BUG (lines 203-207): prior code indexed w by global linear index (garbage
      for every row past the first); corrected to index w by within-row column index.

    * grim_add_rms_norm (lines 210-230): fused add + RMSNorm HIP kernel. y = x +
      residual, then RMSNorm on y. Writes y_out and norm_out. This is a fused norm
      kernel (two ops in one launch).

    * grim_rmsnorm_matmul (lines 258-280): fused RMSNorm + matmul HIP kernel.

    * grim_mla_q_kv_norm_split (lines 358-390): MLA-style Q/KV norm split. RMSNorm on
      q_nope and kv_nope, rope dims copied verbatim. MLA-adjacent norm kernel.

    * grim_silu_mul (lines 151-157): standalone SiLU*up activation kernel.
    * grim_silu_mul_backward (lines 159-172): SiLU backward kernel.
    * grim_softmax (lines 232-247): standalone softmax kernel.
    * grim_embedding (lines 249-256): standalone embedding lookup kernel.

  So grim HAS standalone norm HIP kernels (rms_norm, add_rms_norm fused, rmsnorm_matmul,
  mla_q_kv_norm_split) and standalone rope HIP kernels (rope, rope_yarn) on ROCm.
  The doc's claim "no norm or rope HIP kernel at all" is WRONG. The gap is FUSION DEPTH
  (triple-fused QK-norm+RoPE+KV-insert in one launch), not absence of norm/rope kernels.

*** END VERIFIED STATE ***

What grim ACTUALLY has (corrected):
- grim_rope: standalone plain RoPE HIP kernel (compute_kernels.rs).
- grim_rope_yarn: standalone partial-rotary + YaRN HIP kernel (compute_kernels.rs).
- grim_rms_norm: standalone RMSNorm HIP kernel (compute_kernels.rs), with a
  previously-fixed bug (w indexed by global index -> corrected to within-row column).
- grim_add_rms_norm: fused add+rms_norm HIP kernel (compute_kernels.rs).
- grim_rmsnorm_matmul: fused RMSNorm+matmul HIP kernel (compute_kernels.rs).
- grim_mla_q_kv_norm_split: MLA-style Q/KV norm split HIP kernel (compute_kernels.rs).
- grim_silu_mul, grim_silu_mul_backward, grim_softmax, grim_embedding: other compute
  kernels in compute_kernels.rs.
- Vulkan: rms_norm.comp, rope.comp, silu_mul.comp (separate Vulkan backend).

GAP (corrected): grim has the INDIVIDUAL pieces (RMSNorm, RoPE, SiLU, add+norm fused,
norm+matmul fused, MLA norm split) as standalone or 2-op-fused HIP kernels. What grim
does NOT have is the TRIPLE-FUSED QK-norm + RoPE + KV-insert kernel that vLLM's
fused_qknorm_rope_kernel.cu implements (one launch that does norm(Q), norm(K), RoPE
both, and KV-insert together). The gap is fusion depth, not absence. vLLM also has
DeepSeek-V4 scaling rope and Minimax-M3 variants which grim doesn't have.

PORT DECISION: PORT the TRIPLE-FUSION approach from vLLM (fused_qknorm_rope_kernel.cu)
as an OPTIONAL deeper-fusion kernel, knowing grim already has the component kernels.
The port is "add a deeper-fused option," not "add norm and rope from scratch." Don't
replace grim's existing kernels; add a triple-fused variant alongside them for cases
where the one-launch triple fusion is worth it (e.g. prefill where Q, K, RoPE, and
KV-insert all happen together).

Specifically, port:
(a) The triple-fused QK-norm + RoPE + KV-insert kernel from vLLM's
    fused_qknorm_rope_kernel.cu as a grim HIP kernel. This is the model for the
    deepest fusion. grim's existing norm+rope kernels stay as the component paths;
    the triple-fused kernel is the new "deep fusion" option.
(b) Optionally, the DeepSeek-V4 scaling rope variant (fused_deepseek_v4_qnorm_rope_
    kv_insert_kernel.cu) if grim targets DeepSeek-V4 models.
(c) Optionally, the Minimax-M3 variant (fused_minimax_m3_qnorm_rope_kv_insert_kernel.cu)
    if grim targets Minimax-M3 models.

What NOT to port: the DeepSeek-V4 or Minimax-M3 specifics unless grim targets those
models. The standard triple-fused kernel is the port target.

Priority: P1 (corrected). The gap is smaller than the doc claimed — grim has the
component kernels; the port is the triple fusion, which is a nice-to-have deep-fusion
option, not a missing basic capability.

Skills: rust-gpu (kernel correctness for the triple fusion: the combined norm+rope+
insert math, the shared LDS if any, the wave-aware partitioning if any; correctness
of the fused kernel vs the composed standalone kernels as the parity reference),
rust-ffi (HIP launch for the new triple-fused kernel, the device function helpers),
rocm-kernels (RDNA2/RDNA3 tuning for the triple-fused kernel: block size, num_warps,
LDS usage, the element-wise vs reduction pattern across the three sub-ops).

Mutation-resistant TDD (corrected):
  - RED: source-content test asserts the triple-fused kernel entry exists. If omitted,
    RED. GREEN: present.
  - RED: source-string test asserts the triple-fusion math pattern (norm(Q), norm(K),
    RoPE on both, KV-insert — all in one kernel). If someone ports a non-fused kernel
    (e.g. just norm, or just rope) and labels it "triple-fused," this fails RED. GREEN:
    the literal contains the triple-fusion pattern.
  - RED: COMPOSED-PATH PARITY TEST (new — stronger than CPU oracle alone). The triple-
    fused kernel's output must match the composition of grim's EXISTING standalone
    kernels (grim_rms_norm on Q, grim_rms_norm on K, grim_rope on Q and K, then
    KV-insert) on the same input. This is the key mutation-resistant test: it uses
    grim's own verified kernels as the reference, not an external oracle. If the triple-
    fused kernel is wrong, it diverges from the composed standalone path. If someone
    swaps the triple-fused kernel for a generic kernel, it diverges from the composed
    path. The composed path is the reference because grim's standalone kernels are
    already verified (they have their own tests).
  - RED: edge-case test for the triple-fused kernel with edge-case inputs (e.g. rotary_
    dim < d for partial rotary if supported, or mscale != 1.0 for YaRN). If the kernel
    doesn't handle partial rotary correctly, this catches it.

--------------------------------------------------------------------------------
4. Skinny GEMM (LLMM1 / wvSplitK / wvSplitKrc / wvSplitKQ)  [PORT  P2]
--------------------------------------------------------------------------------

vLLM source: csrc/rocm/skinny_gemms.cu, csrc/rocm/skinny_gemms_int4.cu,
  vllm._custom_ops.LLMM1 / wvSplitK / wvSplitKrc / wvSplitKQ

What vLLM has:
- LLMM1: matrix-vector GEMM (M rows, K cols, N=1 effectively).
- wvSplitK: skinny matmul with CuCount wave-count tuning.
- wvSplitKrc: variant.
- wvSplitKQ: fp8 skinny matmul.
- wvSplitK_int4_g: W4A16 grouped skinny GEMM with per-group scales, optional
  zero_points.
- Excluded on gfx1250 (gfx9/gfx11 ISA unsupported).

What grim has:
- NO skinny GEMM path CONFIRMED BY ABSENCE SEARCH THIS PASS: content search for
  "skinny|LLMM1|wvSplit" in crates/grim-backend-rocm/src/kernels/ returned ZERO
  matches. No skinny-GEMM-named file surfaced. Tentatively accurate.
- BUT NOT POSITIVELY CONFIRMED: this pass checked for absence of the term, not for
  presence of a true matrix-vector or small-M dispatch path inside the existing dense
  GEMM kernels. The claim "NO skinny GEMM path" is unrefuted but unconfirmed.
- grim's GEMM is dense tiled/WMMA/fused-dequant (wmma_gemm, fp8_gemm_rdna4,
  fused_dequant_gemm, q4k/q5k/q6k/q2k/q3k/iq_gemm) -- consistent with the claim.
  GAP: needs a dedicated positive check before trusting as confirmed.
  (e.g. search the existing GEMM kernels for M=1 or small-M dispatch, or for
  matrix-vector product patterns.)

GAP: grim has no matrix-vector or skinny matmul kernel. If grim needs these
(e.g. for certain layer shapes, or for the decoder's Q@K^T when M is small),
vLLM's skinny GEMM is the reference.

PORT DECISION: MAYBE / P2. Port if grim identifies a need for matrix-vector or
skinny matmul. Don't port speculatively.

Specifically, if porting:
(a) LLMM1 (matrix-vector) as a HIP kernel. This is the simplest and most likely
    to be useful. vLLM's skinny_gemms.cu is the reference.
(b) wvSplitK (skinny matmul with wave-count tuning) if grim wants the wave-count
    tuning approach for small-M matmuls.
(c) wvSplitKQ (fp8 skinny) if grim wants fp8 skinny matmul for CDNA.

Priority: P2. Port when grim has a concrete use case, not before.

Skills: rust-gpu (matrix-vector GEMM correctness, the wave-count tuning approach),
  rocm-kernels (RDNA2/RDNA3 skinny matmul tuning: CuCount, the wave-count heuristic).

Mutation-resistant TDD (if ported):
  - RED: source-content test asserts LLMM1 entry exists. RED if omitted.
  - RED: source-string test asserts the matrix-vector math pattern (M rows, K cols,
    N=1 output). If someone ports a dense GEMM labeled "LLMM1", this fails RED.
  - RED: CPU parity oracle. LLMM1 is a matrix-vector product, trivially mirrorable
    on CPU. Given A[M,K], x[K], compute y = A @ x on CPU, compare to HIP output.

--------------------------------------------------------------------------------
5. Fused MoE (W4A16, RDNA3)  [PORT  P0]  *** UPDATED ***
--------------------------------------------------------------------------------

vLLM source: csrc/rocm/moe_q_gemm_rdna3.cu (639 lines)

What vLLM has:
- Fused W4A16 MoE for RDNA3: routing + dequant+dot in one launch, same
  scalar/WMMA split as dense, CAS atomic epilogue, topk_weight mul +
  output_topk reduction.

What grim has:
- Charon: sortless fused MoE dispatch (scalar + WMMA), one launch per
  (token,expert) pair, no host sort, in-register SiLU. charon_wmma.rs,
  charon_backward.rs (FP32 expert-weight backward, SiLU derivative).
- scythe_persistent.rs: opcode 6 = MoE, persistent ring dispatch.
- comm_fuse.rs: SCYTHE-2 WI-6 decomposed P2P fan-in.

UPDATED REASON for porting: grim now targets GPTQ-w4a16 models on RDNA3 (see
section 2). vLLM's fused W4A16 MoE is for GPTQ-w4a16 models. grim's existing MoE
(charon) is for grim's own quant formats (K-quant/IQ-quant/FP8). Both can coexist:
charon serves grim's quant formats; the ported vLLM fused W4A16 MoE serves
GPTQ-w4a16 models. Don't replace charon; add the vLLM-style fused W4A16 MoE as a
separate path for GPTQ models.

PORT DECISION: PORT the approach as a separate GPTQ-W4A16 MoE path. Don't replace
charon.

Specifically, port:
(a) The fused W4A16 MoE kernel from moe_q_gemm_rdna3.cu: routing + dequant+dot in
    one launch, the same scalar/WMMA split as the dense W4A16 GEMM (reuse the
    dequant primitives and GEMM paths from section 2), CAS atomic epilogue,
    topk_weight mul + output_topk reduction.
(b) The host-side launcher that constructs the routing tensors (sorted_token_ids,
    sorted_expert_ids, topk_weights) and launches the fused kernel. vLLM's launcher
    is in the Python wrapper (vllm._custom_ops.moe_gptq_gemm_rdna3); port the
    algorithm as a Rust launcher in grim.

What NOT to port: don't replace charon. charon is for grim's quant formats; the
ported vLLM MoE is for GPTQ-w4a16 models. Both coexist. Don't port the vLLM
Triton MoE paths (triton_moe, triton_cutlass_moe, triton_deep_gemm_moe) -- those
are Triton, not the HIP RDNA3 path grim wants.

Priority: P0 (conditional on section 2 being ported first, because the fused MoE
reuses the W4A16 dequant + GEMM paths).

Skills: rust-gpu (fused MoE dispatch correctness: the routing+dequant+dot fusion,
  the CAS atomic epilogue, the topk_weight mul + output_topk reduction, the
  double-buffered LDS if used), rocm-kernels (RDNA3 tuning for the fused MoE:
  the wave-count, the CAS-loop packed atomic, the precise dequant variant),
  rust-ffi (HIP launch for the fused MoE kernel, the routing tensor construction).

Mutation-resistant TDD:
  - RED: source-content test asserts the fused W4A16 MoE entry exists in
    KERNEL_SOURCE. RED if omitted. GREEN: present.
  - RED: source-string test asserts the fused MoE math pattern (routing arrays:
    sorted_token_ids, sorted_expert_ids, topk_weights; the dequant+dot fusion; the
    CAS atomic epilogue; the topk_weight mul + output_topk reduction). If someone
    ports a non-fused MoE (routing separate from GEMM) and labels it "fused", this
    fails RED. GREEN: the literal contains the fused pattern.
  - RED: CPU parity oracle. The fused W4A16 MoE is mirrorable on CPU: given the
    routing tensors (which tokens go to which experts, the topk weights), the
    W4A16 weights (codes, scales, zeros), and the activations, compute the MoE
    output on CPU (dequant each expert's weights, dot with the assigned tokens,
    apply topk_weight, reduce across experts per token), compare to HIP output.
    The oracle must use the SAME dequant bit-trick and the SAME GEMM approach as
    the kernel, otherwise a swapped dequant/GEMM still passes.
  - RED: cross-path parity test. The fused W4A16 MoE path must produce the same
    output as a non-fused path (routing on CPU, then GEMM per expert separately)
    on the same input. If the fusion introduces a bug (e.g. wrong topk_weight
    application, wrong CAS accumulation), the non-fused path catches it.

--------------------------------------------------------------------------------
6. FP8 KV-cache MFMA in attention  [PORT  P2]
--------------------------------------------------------------------------------

vLLM source: csrc/rocm/attention.cu (FP8 KV path, KV_DTYPE==fp8 branch)

What vLLM has:
- attention.cu supports KV_DTYPE==fp8: k_scale/v_scale, FP8_E4M3_SCALE_TARGET
  normalization, per-warp q_max reduction via warpReduceMax, QK MFMA on fp8 when
  __HIP__FP8MFMA__ (gfx942/gfx950).

What grim has:
- qkv_attention.rs is f16/bf16 only. No FP8 KV. No fp8 QK MFMA.

GAP: grim's attention doesn't support FP8 KV cache. For CDNA targets (gfx942/
gfx950), vLLM's FP8 KV MFMA path is worth porting. For grim's primary gfx1036
(RDNA2), FP8 MFMA is not available, so this is a secondary-arch enhancement.

PORT DECISION: MAYBE / P2. Port for CDNA secondary arch (gfx942/gfx950) if grim
targets those. Not for gfx1036 primary.

Specifically, if porting:
(a) FP8 KV cache support in grim's attention: add KV_DTYPE awareness, k_scale/
    v_scale application, FP8_E4M3_SCALE_TARGET normalization, q_max reduction.
(b) fp8 QK MFMA path gated on gfx942/gfx950 (gfx1036 uses scalar fp8->f32 dequant
    then dot, same as grim's current approach for fp8 weights).

Priority: P2. Secondary-arch enhancement. Don't block primary arch work on this.

Skills: rust-gpu (FP8 KV attention correctness: the scale application, the
  FP8_E4M3_SCALE_TARGET normalization, the q_max reduction, the MFMA path on CDNA),
  rocm-kernels (CDNA FP8 MFMA tuning: the mfma_f32_16x16x32_fp8_fp8 intrinsic,
  the q_max warpReduceMax pattern).

Mutation-resistant TDD (if ported):
  - RED: source-content test asserts the FP8 KV path entry exists. RED if omitted.
  - RED: source-string test asserts the FP8 KV math pattern (k_scale/v_scale,
    FP8_E4M3_SCALE_TARGET, q_max reduction). If someone ports a non-FP8 KV path
    and labels it "FP8 KV", this fails RED.
  - RED: CPU parity oracle. FP8 KV attention is mirrorable on CPU: given Q, K, V
    in FP8 (with scales), compute the attention on CPU (dequant K/V to FP32, apply
    scales, online softmax), compare to HIP output. The oracle must use the SAME
    FP8_E4M3_SCALE_TARGET normalization as the kernel.

--------------------------------------------------------------------------------
7. Norm / RoPE TRIPLE-FUSION kernel  [PORT  P1]  (see section 3)
--------------------------------------------------------------------------------

This is section 3 (corrected). vLLM's fused_qknorm_rope_kernel.cu is the model for
the triple fusion (norm(Q) + norm(K) + RoPE both + KV-insert in one launch).

GRIM STATE (verified this pass against compute_kernels.rs): grim already HAS standalone
and 2-op-fused norm/rope kernels on ROCm — grim_rms_norm, grim_add_rms_norm (fused
add+norm), grim_rope (plain full-rotary), grim_rope_yarn (partial/YaRN), grim_rmsnorm_
matmul (fused norm+matmul), grim_mla_q_kv_norm_split (MLA norm split), grim_silu_mul,
grim_softmax, grim_embedding. The prior doc version said grim had NO norm/rope HIP
kernels at all — that was FALSE. The real gap is FUSION DEPTH: grim lacks the triple-
fused QK-norm+RoPE+KV-insert kernel that vLLM's fused_qknorm_rope_kernel.cu implements.

PORT TARGET (corrected): the triple-fusion kernel, NOT standalone norm/rope (those
exist). Port vLLM's fused_qknorm_rope_kernel.cu as a grim HIP kernel, as an OPTIONAL
deeper-fusion path alongside grim's existing component kernels. The mutation-resistant
test is COMPOSED-PATH PARITY: the triple-fused kernel must match the composition of
grim's EXISTING verified standalone kernels (grim_rms_norm on Q, grim_rms_norm on K,
grim_rope on Q and K, then KV-insert). This is stronger than a CPU oracle because it
uses grim's own verified kernels as the reference.

--------------------------------------------------------------------------------
8. NVFP4 (fp4, Blackwell)  [NO  --]
--------------------------------------------------------------------------------

vLLM source: csrc/libtorch_stable/quantization/fp4/* (nvfp4_scaled_mm_kernels,
  nvfp4_scaled_mm_sm120_kernels, nvfp4_blockwise_moe_kernel, nvfp4_experts_quant,
  nvfp4_quant_entry/kernels, activation_nvfp4 quant fusion) + vllm_xpu_kernels +
  _xpu_ops fp4_gemm

What vLLM has:
- NVFP4 (E2M1 packed) for NVIDIA Blackwell (SM120). AMD has no NVFP4.

What grim has:
- NO NVFP4. grim's lowest-quant is Q2_K / IQ2 family / FP8 / MXFP4.

PORT DECISION: DON'T PORT.

Reason: NVFP4 is NVIDIA Blackwell-only (SM120 / gfx12xx). grim doesn't target
Blackwell. Even if grim did, NVFP4 is a different quantization format that grim
would need to add E2M1 packing for. Not worth it for grim's current target set
(gfx1036 RDNA2, gfx1100 RDNA3, gfx942/gfx950 CDNA).

--------------------------------------------------------------------------------
9. MLA (Multi-Query/Lightweight Attention)  [MAYBE  ??]
--------------------------------------------------------------------------------

vLLM source: vllm.v1.attention.backends.mla.rocm_aiter_mla,
  vllm.v1.attention.backends.mla.aiter_triton_mla, vllm.v1.attention.backends.mla.
  triton_mla

What vLLM has:
- MLA backends for AMD via AITER (rocm_aiter_mla, gfx950 "gluon" padding mode),
  AITER+Triton (aiter_triton_mla), Triton (triton_mla).

What grim has:
- NO MLA kernel.

GAP: grim doesn't have MLA. If grim targets models that use MLA (e.g. DeepSeek-V2/
V3), it needs an MLA kernel.

PORT DECISION: MAYBE. Port if grim targets MLA models. Otherwise skip.

Specifically:
(a) If grim wants MLA: port the approach from vLLM's triton_mla or aiter_triton_mla
    as a HIP kernel. The MLA algorithm (low-rank key compression + shared KV) is
    the thing to port, not the vLLM backend machinery.
(b) If grim doesn't target MLA models: skip. This is model-dependent.

Priority: P2 / conditional. Depends on whether grim targets MLA models.

--------------------------------------------------------------------------------
10. Quickreduce / merge_attn_states  [PORT  P2]
--------------------------------------------------------------------------------

vLLM source: csrc/quickreduce, csrc/libtorch_stable/attention/merge_attn_states.cu

What vLLM has:
- quickreduce: reduce attention states across requests.
- merge_attn_states.cu: merge attention states (prefix + suffix) for cascade
  attention / draft merging.

What grim has:
- NONE.

GAP: grim has no quickreduce or merge_attn_states. If grim supports cascade
attention or draft merging (e.g. speculative decoding with prefix/suffix
merging), these are needed.

PORT DECISION: MAYBE / P2. Port if grim supports cascade attention or draft
merging. Otherwise skip.

Specifically:
(a) quickreduce: a reduce-kernel for attention states. Port the approach.
(b) merge_attn_states: merge prefix + suffix attention states. Port the approach
    if grim does cascade/draft merging.

Priority: P2. Conditional on grim's attention features.

--------------------------------------------------------------------------------
11. ngram embedding index kernel  [PORT  P2]
--------------------------------------------------------------------------------

vLLM source: csrc libtorch_stable ngram kernel + vllm._custom_ops.ngram_compute_n_gram_ids

What vLLM has:
- ngram_compute_n_gram_ids: compute concatenated (offset) n-gram ids for LongCat-
  style models. Writes n_gram_ids of shape [token_num, (ne_n-1)*ne_k].

What grim has:
- NONE.

GAP: grim doesn't have n-gram embedding index kernel. If grim targets LongCat-style
models, it needs this.

PORT DECISION: MAYBE / P2. Port if grim targets LongCat-style models. Otherwise
skip.

Priority: P2. Model-dependent.

--------------------------------------------------------------------------------
12. Triton AMD attention backends (the menu)  [NO  --]
--------------------------------------------------------------------------------

vLLM source: vllm.v1.attention.backends.* (rocm_attn, rocm_aiter_fa,
  rocm_aiter_unified_attn, triton_attn, turboquant_attn, flash_attn, etc.)

What vLLM has:
- A rich 8+ way attention backend menu with priority ordering and per-config
  validation. This is vLLM's v1 attention architecture.

What grim has:
- One QKV attention kernel (qkv_attention.rs) + cross_attention.rs + tree_attention.
- No backend menu.

PORT DECISION: DON'T PORT the menu. Port the kernels.

Reason: grim doesn't need an 8-way menu to start. It needs ONE good paged
attention kernel (section 1). The menu is vLLM's v1 architecture; grim should
build its own attention backend selection when it has multiple backends to choose
between. Porting the menu without the kernels is pointless; porting the kernels
without the menu is fine.

If grim later adds flash attention, turboquant, or MLA backends, then build a
selector. Not now.

--------------------------------------------------------------------------------
13. Vulkan  [N/A  --]
--------------------------------------------------------------------------------

vLLM: no Vulkan backend. vllm_xpu_kernels / _xpu_ops is Intel XPU (SYCL), not
Vulkan.

grim: genuine Vulkan compute backend (grim-backend-vulkan, ~4482 lines lib.rs,
SPIR-V shaders compiled at build time + embedded as include_bytes!, full kernel
set coverage).

PORT DECISION: N/A. vLLM has nothing to port here. grim already has Vulkan;
vLLM doesn't. This is a grim advantage, not a port target.

--------------------------------------------------------------------------------
14. Summary: port priority order (UPDATED)
--------------------------------------------------------------------------------

Priority order for porting from vLLM to grim (AMD ROCm):

P0 -- MUST PORT:
  1. PagedAttention EXTENSIONS (section 1, corrected). grim already HAS a real paged
     attention kernel. The P0 work is to EXTEND it: add MFMA QK gated on gfx1100+
     (RDNA3), add FP8 KV cache for CDNA (gfx942/gfx950). The base paging already
     exists; the gap is feature breadth, not presence.
  2. W4A16 GPTQ GEMM (dense + WMMA) for RDNA3 (section 2). grim targets GPTQ-w4a16
     models on RDNA3. Port the dequant primitives, scalar GEMM, WMMA GEMM, and K-split
     heuristic from vLLM's q_gemm_rdna3.cu + q_gemm_rdna3_wmma.cu. Watch naming
     collision with any offline GPTQ calibration file.
  3. Fused W4A16 MoE for RDNA3 (section 5). Add as a separate GPTQ-W4A16 MoE path
     (reuses section 2's dequant + GEMM). Don't replace charon.

P1 -- PORT SOON:
  4. Triple-fused QK-norm + RoPE + KV-insert kernel (section 3, corrected). grim
     already HAS standalone RMSNorm (grim_rms_norm), fused add+rms_norm
     (grim_add_rms_norm), plain RoPE (grim_rope), YaRN/partial RoPE (grim_rope_yarn),
     rmsnorm_matmul, mla_q_kv_norm_split, SiLU (grim_silu_mul), softmax — all real
     HIP kernels in compute_kernels.rs. The P1 work is to ADD a deeper triple-fused
     option (vLLM's fused_qknorm_rope_kernel.cu as the model), NOT to add norm/rope
     from scratch. The mutation-resistant test is composed-path parity: the triple-
     fused kernel must match the composition of grim's existing verified standalone
     kernels.
  5. (Skipped — section 3 / section 7 were the same; section 3 now covers the
     triple-fusion port. Section 7's "standalone RMSNorm + RoPE kernels are the port
     target" is obsolete — those kernels exist. Remove or fold into section 3.)

P2 -- PORT LATER / CONDITIONAL:
  6. MFMA for QK (gfx1100+ secondary arch) — now part of section 1's extensions.
  7. FP8 KV cache attention (gfx942/gfx950 secondary arch) — now part of section 1's
     extensions.
  8. Skinny GEMM / LLMM1 (section 4). Port if grim needs matrix-vector or skinny
     matmul. Claim unconfirmed — run a dedicated positive check first.
  9. Fused norm+rope kernel — obsolete; replaced by triple-fusion port (section 3).
  10. MLA kernel (section 9). Port if grim targets MLA models.
  11. Quickreduce / merge_attn_states (section 10). Port if grim supports cascade
      attention or draft merging.
  12. ngram embedding index kernel (section 11). Port if grim targets LongCat-style
      models.

NO -- DON'T PORT:
  - NVFP4 (section 8). NVIDIA Blackwell-only; grim doesn't target Blackwell.
  - Triton AMD attention backend menu (section 12). Port kernels, not the menu.
  - Vulkan (section 13). vLLM has none; grim already has it.
--------------------------------------------------------------------------------

These are grim's strengths relative to vLLM. Don't port vLLM's approach here --
grim already wins:

- K-quant/IQ-quant (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/IQ2/3/4): vLLM has zero coverage.
  grim owns this. Don't port from vLLM (there's nothing to port).
- Sortless MoE dispatch + persistent ring (charon + scythe_persistent): vLLM's MoE
  is host-sorted. grim's is sortless device dispatch. Don't port vLLM's MoE -- but
  ADD the vLLM-style fused W4A16 MoE as a separate path for GPTQ models (section 5).
- Native HIP FP8 GEMM ownership (wmma fp8 + tiled + fused dequant fp8): vLLM's AMD
  FP8 is cutlass-via-compat + Triton. grim has native HIP FP8 GEMM. Don't port
  vLLM's FP8; grim's is more owned.
- Native HIP MXFP4/8 dequant: vLLM's is Triton. grim's is native HIP. Don't port.
- Vulkan backend: vLLM has none. grim has it. Don't port (nothing to port).
- RWKV: vLLM has no RWKV AMD kernel. grim has rwkv.rs + Vulkan. Don't port.
- KV dequant attention (own mechanism): vLLM has fp8-KV-in-MFMA; grim has
  dequant-on-read. Different mechanisms; both valid. Don't port -- grim's is fine.
- MWRA (wave64) primary arch optimization: vLLM's attention.cu is wave-aware but
  targets gfx11/gfx9 primarily. grim's qkv_attention.rs is wave64-optimized for
  gfx1036. Don't lose this.

--------------------------------------------------------------------------------
16. Porting process (how to port, not what)
--------------------------------------------------------------------------------

For each PORT item above, the process is:

1. Read the vLLM source file(s) listed in that section.
2. Understand the algorithm (not the C++ details).
3. Write a grim HIP kernel as `pub const KERNEL_SOURCE: &str = r#\"...\"#` in the
   appropriate grim-backend-rocm/src/kernels/<name>.rs file.
4. Add a Rust launcher function in the same file or in the device module.
5. Add the four mutation-resistant test classes (see section 18) for the new kernel.
6. Add a Rust benchmark (cargo bench or cfgrind) to verify the port is faster than
   the grim baseline (or at least not slower).
7. Wire the new kernel into the device dispatch (RocmDevice or the kernel module).

For items marked NO: skip. No code written.

For items marked MAYBE: defer the decision until grim has a concrete need (model
support, feature request). Don't port speculatively.

--------------------------------------------------------------------------------
17. Files referenced (source of truth for ports)
--------------------------------------------------------------------------------

vLLM AMD ROCm C++ kernels (csrc/rocm/):
- attention.cu          -> PagedAttention, MFMA QK, FP8 KV attention (P0, P2)
- q_gemm_rdna3.cu       -> W4A16 GPTQ dense (P0, UPDATED)
- q_gemm_rdna3_wmma.cu  -> W4A16 GPTQ WMMA (P0, UPDATED)
- moe_q_gemm_rdna3.cu   -> Fused W4A16 MoE (P0, UPDATED)
- qdq_4_rdna3.cuh       -> W4A16 dequant primitives (P0, reference for dequant bit-trick)
- skinny_gemms.cu       -> LLMM1 / wvSplitK / wvSplitKrc / wvSplitKQ (P2)
- skinny_gemms_int4.cu  -> wvSplitK_int4_g (P2)

vLLM libtorch_stable (CUDA/HIP, secondary reference):
- fused_qknorm_rope_kernel.cu               -> norm+rope fusion (P1, P2)
- fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu -> DeepSeek-V4 rope (P2, conditional)
- fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu -> Minimax-M3 rope (P2, conditional)
- rms_norm op / fused_add_rms_norm op       -> RMSNorm (P1)
- selective_scan_fwd.cu                     -> Mamba selective scan (grim already has this)
- merge_attn_states.cu                      -> merge attention states (P2)
- ngram kernel                              -> ngram embedding index (P2)

vLLM Python wrappers (signatures, not kernels):
- vllm._custom_ops.py        -> paged_attention_rocm, LLMM1, wvSplitK*, rotary_embedding,
                                rms_norm, fused_add_rms_norm, ngram_compute_n_gram_ids,
                                moe_gptq_gemm_rdna3 (P0, UPDATED)
- vllm._rocm_C               -> the HIP extension that wraps csrc/rocm/*.cu
- vllm._xpu_ops.py           -> Intel XPU (SYCL), NOT AMD, NOT Vulkan -- ignore for this task

vLLM v1 attention backends (backend selection, not kernels):
- vllm.v1.attention.backends.rocm_attn.py  -> RocmAttentionBackend (paged attention wrapper)
- vllm.v1.attention.backends.mla/*.py      -> MLA backends (MAYBE)
- vllm.v1.attention.backends/*.py          -> the menu (NO -- port kernels, not menu)

grim sources (what grim already has, for comparison):
- crates/grim-backend-rocm/src/kernels/qkv_attention.rs       -> Phase-1 QKV (needs paging)
- crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs -> fused dequant GEMM (grim's quant)
- crates/grim-backend-rocm/src/kernels/charon.rs             -> sortless MoE (keep; add vLLM-style
                                                                fused W4A16 MoE alongside)
- crates/grim-backend-rocm/src/kernels/charon_wmma.rs        -> Charon WMMA (keep)
- crates/grim-backend-rocm/src/kernels/wmma_gemm.rs          -> f16 WMMA GEMM (reference for
                                                                WMMA setup idiom, reuse for GPTQ WMMA)
- crates/grim-backend-rocm/src/kernels/scythe_persistent.rs -> persistent ring (keep)
- crates/grim-backend-vulkan/src/lib.rs                      -> Vulkan backend (vLLM has none)

--------------------------------------------------------------------------------
18. MUTATION-RESISTANT TDD FOR ALL KERNEL PORTS
--------------------------------------------------------------------------------

Every PORT section in this doc includes a "Mutation-resistant TDD" subsection with
the four test classes for that kernel. This section is the general framework; the
per-kernel subsections restate it with the specific values.

WHY MUTATION-RESISTANT: a normal test that checks "output is close to a reference"
can be satisfied by a kernel that does the wrong thing but happens to be close enough
on the tested inputs. A mutation-resistant test is one where if you mutate the kernel
in a way that changes its behavior (swap the algorithm, drop a step, change the math),
the test fails. The four test classes together provide this:

CLASS 1 -- SOURCE-CONTENT TEST (RED/GREEN on presence):
  Asserts that the expected kernel entry symbol(s) exist in KERNEL_SOURCE as a
  string. This is the lowest-bar test: if the kernel is omitted, this fails.
  Example: `assert!(KERNEL_SOURCE.contains("grim_qkv_attention_paged"))`.
  Mutation it catches: omitting the kernel entirely.
  Mutation it does NOT catch: including a stub kernel with the right name but wrong
  body. That's what class 2 is for.

CLASS 2 -- SOURCE-STRING TEST (RED/GREEN on math pattern):
  Asserts that the KERNEL_SOURCE string contains specific math/algo patterns that
  identify the CORRECT algorithm, not just any kernel with the right name. These
  patterns are chosen so that a plausible wrong kernel (same name, different math)
  would NOT contain them.
  Example for paged attention: assert the literal contains "block_tables",
  "page_size", "physical_token_idx = entry.block_id * page_size + t" -- a non-paged
  attention kernel would not contain these.
  Example for W4A16 dequant: assert the literal contains the GPTQ bit-trick
  (0x64006400, half2(1024+q, 1024+q*16)) -- a generic 4-bit unpack would not.
  Mutation it catches: swapping the algorithm for a different one with the same name.
  Mutation it does NOT catch: a kernel that contains the right patterns but computes
  them wrong (e.g. wrong K-split heuristic that still mentions "k_split"). That's
  what class 3 is for.

CLASS 3 -- CPU PARITY ORACLE TEST (RED/GREEN on numeric correctness vs reference):
  A pure-Rust (or pure-Python, or pure-CPU) reference implementation of the SAME
  algorithm the kernel implements. The HIP kernel is launched, output read back via
  hipMemcpyDtoH after hipDeviceSynchronize, and compared to the reference within
  tolerance. The reference must be the CORRECT algorithm, not an approximate one.
  If the kernel computes the wrong thing, the oracle catches it.
  Example for W4A16 GEMM: CPU reference = dequant W4A16 weights using the SAME
  bit-trick as the kernel, then FP32 GEMM of dequantized weights vs input A, compare
  to HIP output. If the kernel's dequant bit-trick is wrong, the CPU reference (using
  the correct bit-trick) diverges.
  Example for paged attention: CPU reference = simulate vLLM's paged attention on CPU
  (block_table walk, page decomposition, online softmax), compare to HIP output. If
  the kernel's page walk is wrong, the oracle catches it.
  Mutation it catches: any algorithmic error that changes the numeric output.
  Mutation it does NOT catch: a kernel that matches the CPU reference on the tested
  inputs but is wrong on untested inputs (e.g. only tested on M=1). That's what
  class 4 is for. Also: if the CPU reference itself is wrong (mirrors the kernel's
  bug), both pass -- so the CPU reference must be independently verified (e.g. against
  vLLM's output or a known-correct implementation).

CLASS 4 -- METAMORPHIC / CROSS-PATH TEST (RED/GREEN on behavioral invariants):
  A test that checks a behavioral invariant that must hold for the correct algorithm
  but would not hold for a wrong one, using inputs that exercise the dimension that
  distinguishes the correct algorithm from plausible wrong ones.
  Example for paged attention: use an input where seq_lens span page boundaries
  (so paging matters). A non-paged kernel would produce different output on this
  input than the paged kernel. The test asserts the paged kernel's output matches
  vLLM's paged output on this input.
  Example for W4A16: cross-path parity -- the scalar GEMM path and the WMMA GEMM
  path must produce the same output on the same input. If the WMMA path has a layout
  bug, the scalar path catches it. (Paired with class 1 to ensure the WMMA path
  actually contains the WMMA intrinsic, not just a scalar path labeled "WMMA".)
  Example for RMSNorm: test with weight=None (if supported) -- a kernel that assumes
  weight is always present would segfault or produce wrong output. vLLM's LTX-Video
  pitfall is exactly this.
  Mutation it catches: errors that only manifest on specific input shapes or edge
  cases, not on the generic inputs used for class 3.

TIERED TOLERANCE: the tolerance in class 3 must be tight enough to catch real errors
but loose enough to allow fp16/bf16/FP8 rounding. For FP32 accumulators with fp16
inputs, max_abs < 1e-3 is usually fine. For FP8 dequant, the tolerance depends on the
format's precision (Q4_K is coarse; FP8 E4M3 is finer). The tolerance must be stated
explicitly in each test, and must be verified against vLLM's output or a known-correct
reference, not chosen arbitrarily.

WHEN TO ADD EACH CLASS:
  - Class 1 and 2 are added at the same time as the kernel literal (RED before the
    literal exists, GREEN after). They're fast (string assertions on the source).
  - Class 3 is added once the kernel is JIT-compilable and launchable. If no GPU is
    available (sandbox), class 3 is written as a pure-Rust CPU parity test that
    compares the CPU reference to itself (trivially green) -- the test structure is
    in place, and it becomes a real GPU test when a GPU is available. The CPU reference
    function is pure and testable without a GPU.
  - Class 4 is added once class 3 is green, using inputs that exercise the behavioral
    invariant. Class 4 often requires specific input shapes that may need a GPU to
    exercise fully; in the sandbox, class 4 is written with the CPU reference + the
    kernel's CPU fallback path (if the kernel has a scalar fallback like wmma_gemm.rs
    does), so the cross-path test can run on CPU.

EXISTING GRIM PRECEDENT: grim-backend-rocm/src/kernels/wmma_gemm.rs and
charon_wmma.rs already use class 1 + class 2 (source-content + source-string tests).
They do NOT yet use class 3 (CPU parity oracle) or class 4 (metamorphic/cross-path)
-- those are the new requirements for this port. The existing tests are necessary but
not sufficient for mutation resistance; they catch omission and wrong-name-wrong-body,
but not wrong-math-that-looks-right or wrong-on-untested-inputs.

CPU PARITY ORACLE PATTERN (the key new piece):
  For each ported kernel, write a pure-Rust function that computes the same result on
  CPU. This function:
  - Takes the same inputs as the kernel (represented as Rust slices/vectors).
  - Computes the result using the SAME algorithm as the kernel (same bit-tricks, same
    math, same reductions), not an approximate algorithm.
  - Is independently verified against vLLM's output (if vLLM's output is available) or
    against a known-correct reference, so the oracle itself is trusted.
  - Is tested on CPU without a GPU (pure Rust, deterministic, no HIP calls).
  The HIP kernel is then compared to this CPU reference. If the kernel is wrong, the
  comparison fails. If the CPU reference is wrong, both pass -- so the CPU reference
  must be independently verified.

  For kernels that have a scalar fallback path (like wmma_gemm.rs's gfx1036 fallback),
  the CPU reference can be compared to the scalar fallback path on CPU (if the fallback
  is available as a separate compilable unit) to provide a cross-check without a GPU.

MUTATION EXERCISE (optional but recommended): after each kernel's four test classes
are green, mutate the kernel source in a way that changes its behavior (e.g. remove
the K-split heuristic, swap the dequant bit-trick for a generic unpack, remove the
paged page walk) and confirm that at least one test class fails RED. If no test fails,
the tests are not mutation-resistant and need strengthening. This exercise is done in
the source literal (not on a GPU -- just editing the string and re-running the source
tests), so it can be done in the sandbox.

--------------------------------------------------------------------------------
19. Skills referenced (why each is cited)
--------------------------------------------------------------------------------

rust-ffi: Cited in every PORT section that adds a new kernel launch. Why: the new
  kernel is a HIP literal that must be JIT-compiled via hiprtc or loaded via
  hipModuleLoad, then launched via hipModuleLaunchKernel. The FFI skill covers the
  HIP runtime FFI (hipMalloc/hipFree/hipMemcpy/hipModule* functions), the status
  checking pattern (hip_check), the dlopen-vs-link-time decision for which ROCm
  runtime to load (grim uses side-by-side .rocm-N trees), and the rocblas FFI if
  the kernel delegates to rocblas. The FFI skill's ROCm section is the reference for
  how to bind these C functions safely in Rust.

rust-gpu: Cited in every PORT section. Why: each ported kernel is a GPU kernel, and
  rust-gpu is the skill for GPU kernel correctness in grim. It covers the LDS budget
  discipline, the wave size (W64 vs W32) adaptation, the parity-vs-reference
  discipline (class 3 CPU parity oracle), and the device-gated compilation pattern
  (the #if defined(__gfx1100__) guards that grim already uses in wmma_gemm.rs and
  charon_wmma.rs). The skill ensures the ported kernel is correct on the target arch
  before optimizing for performance.

rocm-kernels: Cited in every PORT section that targets AMD specifically. Why: the
  skill covers AMD GPU kernel tuning (RDNA2 wave64 constraints, RDNA3 wave32 layout,
  CDNA MFMA intrinsics, the v_dot2 intrinsic, the CAS-loop packed atomic, the precise
  dequant variant, the LDS sizing for the wave-aware partitioning pattern vLLM uses).
  For the W4A16 GPTQ port (section 2), rocm-kernels is the primary skill because the
  entire port is RDNA3-specific. For the paged attention port (section 1), rocm-kernels
  covers the wave-aware partitioning pattern that vLLM's LL4MI kernel uses and grim's
  qkv_attention.rs already mirrors.

rust-testing: Cited in section 18 (mutation-resistant TDD). Why: the skill covers
  Rust test organization (unit tests inline with #[cfg(test)], integration tests in
  tests/, doc tests), the assertion patterns (assert_eq!, assert!, should_panic),
  and the test discipline (one assertion per test, edge cases, descriptive names,
  test isolation, mock dependencies). The mutation-resistant TDD framework in section
  18 builds on this: the four test classes are Rust tests following this discipline.

rust-unit-testing: Cited in section 18 (mutation-resistant TDD). Why: the skill
  covers advanced Rust test patterns that make the mutation-resistant tests clearer
  and more maintainable: rstest for parameterized test cases (e.g. the CPU parity
  oracle can be parameterized over multiple input shapes), googletest for matcher
  assertions (e.g. the tolerance comparison `assert_that!(output, roughly_eq!(reference,
  1e-3))`), pretty_assertions for diff-friendly equality, insta for snapshot testing
  (e.g. snapshot the CPU reference output for a canonical input, so regressions are
  visible). The skill's guidance on test helper design (split helpers into extractor +
  predicate + assertion wrapper) applies to the CPU parity oracle helpers.

--------------------------------------------------------------------------------
Report done. Archives: vllm-main csrc/rocm/ (attention.cu, q_gemm_rdna3.cu,
q_gemm_rdna3_wmma.cu, moe_q_gemm_rdna3.cu, qdq_4_rdna3.cuh, skinny_gemms.cu,
skinny_gemms_int4.cu, ops.h, torch_bindings.cpp), vllm libtorch_stable (fused_qknorm_
rope_kernel.cu, fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu, fused_minimax_m3_
qknorm_rope_kv_insert_kernel.cu, rms_norm kernels, selective_scan_fwd.cu, merge_attn_
states.cu, ngram kernel), vllm _custom_ops.py + _xpu_ops.py + platforms/rocm.py +
v1 attention backends, vllm quantization kernel dirs, vllm Triton kernel dirs,
grim-backend-rocm kernel sources re-read (qkv_attention.rs, fused_dequant_gemm.rs,
wmma_gemm.rs, charon_wmma.rs, mod.rs), grim-backend-vulkan lib.rs (Vulkan parity
check). Skills loaded: rust-ffi, rust-gpu, rocm-kernels, rust-testing, rust-unit-testing.
No external sources used.
