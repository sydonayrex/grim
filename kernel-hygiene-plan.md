# Kernel hygiene implementation plan

Three follow-ups from the vLLM `qdq_4_rdna3.cuh` comparison. All refs verified at
source.

1. Batch-widen the 4-bit/K-quant nibble extraction in charon's `iqk_weight`
   using the zero-FMA-cost bit-trick pattern (`dequant_4bit_8_bf16_q_only`-
   style), scoped to grim's existing fp32 accumulation. No bf16 packed-FMA
   angle — that is not grim's bottleneck.
2. Separately evaluate a packed/coalesced atomic epilogue for
   `grim_moe_fused_grouped`'s output accumulation, inspired by
   `atomic_add_pk4_*` but redesigned for fp32 output, not ported directly.
3. Split the `fp8_native: bool` capability pillar so RDNA4 (OCP e4m3fn) and
   CDNA3 (e4m3fnuz) are no longer conflated.
4. Add a standalone per-token × per-channel scale+bias epilogue kernel after the
   rocBLAS GEMM (the only structurally-possible path on classic rocBLAS).

================================================================================
0. SCOPE / RULES
================================================================================

- All 4 dequant sites must stay bit-identical: grim-quant CPU oracle, standalone
  GPU dequant kernel, fused dequant-GEMM forward kernel, ROCm equivalent
  (grim-backend-correctness Pattern A). A fused-only change that passes CPU tests
  but breaks on-device parity is the #1 silent-gibberish bug. Verify with a
  GPU-kernel-vs-oracle test per change.
- Every new `extern "C" __global__` kernel follows the repo convention: HIP
  source literal (`pub const KERNEL_SOURCE: &str = r#"..."#`), `#[repr(C)]`-safe
  args, pointer null checks before deref, no panic across the FFI boundary
  (rust-ffi-grim §1).
- Block dims are multiples of the wave size (32 on RDNA, 64 on CDNA); kernels
  read `warpSize` at runtime, never hardcode (rocm-hip-kernels / WR wave rule).
- fp8 stays gated on the arch string; never on type availability
  (rocm-hip-kernels / rust-ai-ml-inference-guide Action 9).

================================================================================
1. Batched 4-bit/K-quant nibble widening in `iqk_weight` (fp32)  [implement]
================================================================================

Scope (this is the DECODE half of the original bf16-widen idea, narrowed):
- Batch-widen the 4-bit/K-quant nibble extraction in `iqk_weight` using the
  zero-FMA-cost bit-trick pattern (`dequant_4bit_8_bf16_q_only`-style): bit-
  extract every nibble of a loaded `qs[]` byte and produce the widened integer
  value with NO per-element FMA spent just to create the widened form.
- Scoped to grim's existing fp32 accumulation. `iqk_weight` feeds the fp32
  `gate` / `up` / `acc` chain and stays fp32 throughout.
- NO bf16 packed-FMA angle. We are NOT adopting a packed-pair `__hfma2` widen
  pipeline or a bf16 accumulator: the vLLM reference widens to bf16 because its
  GEMM consumes bf16, whereas grim's fused MoE contraction consumes fp32. Here
  the trick lands as the in-register integer nibble value, followed by the
  EXISTING fp32 `fmaf` — no bf16 staging anywhere.
- The OUTPUT-ACCUMULATOR question (packed/coalesced atomic epilogue) is a
  SEPARATE item — section 2 — and does not ride along in this change.

Sources (all verified):
- charon.rs:556 `iqk_weight` device fn; fmt==7 Q4_K branch at charon.rs:641-658.
  The Q4_K branch is strictly per-element: f16_to_f32 twice (dd, dmin), the
  k<4-vs-else scale/min unpack for (sc1,m1,sc2,m2), nibble extract, then two
  separates. Called from the hot MoE triple loop charon.rs:778-784 — one
  `iqk_weight` call per (i) per (j) per (gate|up), and per j for down. Every
  call re-does the scale unpack + f16 widen.
- CPU oracle grim-quant:596-656 `dequant_q4k` already hoists correctly:
  (sc1,m1)/(sc2,m2) unpacked once per 64-weight sub-block, folded to
  `d1 = d*sc1`, `m1_val = min*m1`, then a 32-weight loop applies `d1*q - m1_val`.
  This is the reference structure to mirror — the CPU oracle is the ground truth.

What we port = the TECHNIQUE only:
- Hoist the per-sub-block work out of the per-element hot path, exactly like the
  oracle does: load dd, dmin once; unpack (sc1,m1,sc2,m2) once per sub-block
  (hoists the k<4 branch); precompute d1/d2 and m1v/m2v once; per-element is a
  single nibble extract + `fmaf(d, q, -m)` — no f16 widen, no branch.
- Batch the nibble decode: one `qs[l]` byte load yields 2 weights (low nibble
  for hi==0, high nibble for hi==1). Decode both from the same byte instead of
  two separate indexed loads.
- Keep the two-tier super-block scale/min structure intact. Do NOT reorganize the
  weight bytes to a linear layout; the vLLM trick's flat `(q-zero)*scale` math
  has no valid analogue while Q4_K needs a per-sub-block scale/min lookup, and
  rewriting the extraction = redesigning the format (out of scope).

Precise change at charon.rs fmt==7 branch:
```
// per sub-block (hoisted):
  float d1 = dd * (float)sc1;  float m1v = dmin * (float)m1;
  float d2 = dd * (float)sc2;  float m2v = dmin * (float)m2;
// per element (in the 32-loop over l):
  int lo = qs[q_off + l] & 0x0F;                       // hi==0
  val = fmaf(d1, (float)lo, -m1v);                     // 1 FMA, no widen
  int hi = qs[q_off + l] >> 4;                         // hi==1
  val = fmaf(d2, (float)hi, -m2v);
```
Equivalent restructuring for Q5_K (fmt==8, charon.rs:659-686, same oracle hoist
in grim-quant dequant_q5k) and Q6_K (fmt==9) can ride along in the same change,
but Q4_K is the priority (it is the kernel actually exercised).

Files touched:
- crates/grim-backend-rocm/src/kernels/charon.rs (fmt==7 branch, maybe 8/9).
- Any fused forward + standalone dequant twin in the same crate for parity.

Acceptance:
- New batched charon path bit-equal (or <1 ulp) to grim-quant `dequant_q4k`
  oracle for the same 144-byte block bytes. Add/keep the GPU-parity test: upload a
  known Q4_K block to device, run the fused MoE dequant path, read back, compare
  to the oracle (same recipe as grim-backend-correctness Pattern A).
- Fewer per-element ops in the hot loop: no per-element f16_to_f32, no per-element
  scale branch, half the qs byte loads (2 nibbles/byte).
- `cargo test -p grim-backend-rocm --lib charon` green; real model run
  unchanged output (Claude/llama-style probe, GRIM_BACKEND set).

================================================================================
2. Packed/coalesced fp32 atomic epilogue for moe output  [evaluate]
===============================================================================

Separate item (split out of section 1's original scope). This is a RESEARCH /
EVALUATION task, not a port: decide whether `grim_moe_fused_grouped*`'s output
accumulation should adopt a packed/coalesced atomic add pattern — inspired by
the `atomic_add_pk4_*` family (a shared pattern in LLM kernel land: pack several
independent accumulator lanes into one wider atomic RMW so N adds cost ~1 RMW,
then unpack on read). Redesigned for grim's fp32 `out` buffer, NOT ported:
`atomic_add_pk4_*` is written around narrow lanes (int8/int16/bf16 pairs); grim
accumulates plain fp32, so the packing idiom does not carry over 1:1.

Current state (all verified at source):
- charon.rs:786-788 (`grim_moe_fused_grouped_iqk` output loop): for every h,
  `atomicAdd(out + (tok*hidden + h), routed_scaling_factor * w * as * acc)`
  inside the h-loop — i.e. ONE fp32 32-bit atomicAdd per (token, hidden) element,
  from each routed (token, expert) row. Since a token is routed to top_k
  experts, the same `out[tok*hidden+h]` address is hit up to top_k times across
  different s entries, all within the same stream.
- The analogous printf-style all-variants epilogue is shared: fp8 (charon.rs:252),
  mxfp4 (:349), mxfp8 (:410), q80 (:483), iqk (:742) all end the same way.

How `atomic_add_pk4_*` works (the reference, not a source to copy):
- It requires output lanes to be packable into a 32/64-bit word, e.g. 4×int8 or
  2×bf16. RMW is done on the wider word with a compare-exchange; lanes are
  unpacked afterward. The win comes from cutting the number of RMWs when the
  atomic itself is on the critical path (high contention on the same cache line).
- For grim fp32 output, lanes are 32 bits — a natural pk4 word would be 128-bit
  (4×fp32 via a 128-bit CAS), or a pk2 (2×fp32 via 64-bit). Grim's `out` row is
  contiguous per token, so a thread's h-loop already has adjacent addresses to
  pack. Whether CAS-based wide atomics beat 32-bit atomicAdd on RDNA/CDNA needs
  a device benchmark — this is exactly what the evaluation must measure.

Evaluation plan (evidence-first):
1. Grep the actual epilogue occurrences and confirm all six variants share the
   identical scalar atomicAdd (done above; confirm no variant has a divergent epilogue).
2. Microbenchmark on target device (RDNA gfx110x/1200 and CDNA gfx942 if both
   are testable): 32-bit `atomicAdd` vs 64-bit CAS fp32x2 vs 128-bit CAS fp32x4,
   with the real contention shape — same `out[h]` written by top_k simultaneous
   rows. Measure RMW throughput + contention stalls, not wallclock noise.
3. Only adopt a packed/coalesced epilogue if it measurably beats the scalar
   atomicAdd on the shape the kernel actually runs (token-count × hidden where
   hidden ≈ 14336 for real grim models). Otherwise recording "keep scalar
   atomicAdd" IS the deliverable.
4. If a variant wins, implement it conservatively: `out` stays fp32, the packed
   lanes are (h, h+1) / (h..h+3) pairs within the same token row, and a
   thread-safe read-back parity test (grim-backend-correctness Pattern A:
   CPU oracle vs GPU result) MUST pass bit-equal on the accumulate path.

Scope fence:
- This is NOT a change to the decode math (section 1's). It only restructures the
  final write. If it lands, it must not perturb the inner-loop `gate`/`up`/`acc`
  values, and the KAT oracle must remain the same.
- No format change to `out`; packed lanes are an ephemeral RMW trick, the buffer
  contents stay plain fp32.

Files touched:
- crates/grim-backend-rocm/src/kernels/charon.rs (epilogue only, if adopted).
- crates/grim-backend-rocm/tests/* (parity / golden test if landed).

Acceptance:
- A written decision (adopt / reject) with the measured microbenchmark numbers
  recorded in this plan or an adjacent ADR-style note.
- If adopted: `cargo test -p grim-backend-rocm --lib charon` green and goldens
  pass; no decode-math change bundled in the same commit.

===============================================================================
3. Split fp8_native capability (OCP-FN vs FNUZ)  [implement]
================================================================================

Sources (all verified):
- quantization.rs:156 `arch_capability`: `GcnArch::RDNA4 | GcnArch::CDNA3` → one
  arm setting `fp8_native: true`. QuantCapability:116-123 has a single bool,
  no e4m3fn/e4m3fnuz distinction anywhere in QuantMode/QuantCapability.
- GcnArch already separates the two arches correctly at quantization.rs:14-19.
- Consumers of the bool today: accel_features.rs mfma_dispatch:37 / wmma_dispatch:72
  (both just `QuantMode::Fp8Native` match arms), lib_internal_tests.rs:396,405,
  self_tests. Confirmed: no *production* dispatch consumes fp8_native yet — only
  tests — so this is a latent inconsistency, not a live bug. But charon's W8A8
  path is heading straight at it; fix before any W8A8 dispatch lands.

Why it matters:
- RDNA4 (gfx1200) native fp8 = OCP e4m3fn. CDNA3 (gfx942) native fp8 = e4m3fnuz.
  One bool cannot express the difference, and it is exactly the axis a W8A8 GEMM
  must branch on (packed-code decode differs; MFMA predicate differs).
- fp8 hardware rule (rocm-hip-kernels, rust-ai-ml-inference-guide): fp8 MFMA is
  gfx1200+ hardware; on gfx1036/gfx110x the *type* exists but is emulated and
  slower than f16. Keep gating on the arch string.

Change — split the capability OUTPUT, not the arch detection:
- Replace/extend `QuantCapability.fp8_native: bool` with a distinct format field,
  e.g.:
```
pub enum Fp8NativeFormat { None, OcpFn, Fnuz }

pub struct QuantCapability {
    ...
    fp8_native: Fp8NativeFormat,   // RDNA4 -> OcpFn, CDNA3 -> Fnuz, else None
}
```
- arch_capability: RDNA4 arm -> `Fp8NativeFormat::OcpFn`; CDNA3 arm ->
  `Fp8NativeFormat::Fnuz`; every other arm -> `None`. RDNA4|CDNA3 merged arm
  must be split into two arms.
- `supports(QuantMode::Fp8Native)` -> `fp8_native != Fp8NativeFormat::None`, so
  existing call sites (accel_features, lib_internal_tests) keep compiling and the
  two existing tests keep passing.
- Add the precise variants if W8A8 dispatch wants them later:
  `QuantMode::Fp8NativeOcpFn(Fp8NativeFormat::OcpFn)` /
  `QuantMode::Fp8NativeFnuz(Fp8NativeFormat::Fnuz)` — do that in the W8A8 change,
  not here.

NAMING CROSS-CHECK (flagged during review, confirmed):
- "OCP" already means OCP *Microscaling* (MXFP4/MXFP8, the Jay/Magpie tiers) all
  over the same crate and grim-quant: charon.rs:242,326,407; grim-quant:1149,1215.
  The new enum's OCP = OCP's *element format* (e4m3fn). Different axis entirely.
  Use the unambiguous names `OcpFn` / `Fnuz` (or `Fp8NativeOcpFn` /
  `Fp8NativeFnuz` for QuantMode), NOT a bare `Ocp`. The enum variants above are
  named to avoid the collision.

Files touched:
- crates/grim-backend-rocm/src/quantization.rs (enum, struct, arch_capability,
  Display, supports, self_tests).
- crates/grim-backend-rocm/src/device/accel_features.rs:37,72 (match arms, only if
  touched by the enum rename).
- lib_internal_tests.rs:396,405 keep asserting `supports(Fp8Native)` truths.

Acceptance:
- `cargo test -p grim-backend-rocm --lib quantization` and `--lib accel_features`
  green with the split; RDNA4 -> OcpFn, CDNA3 -> Fnuz asserted.
- No behavior change for anything that consumed the bool.

================================================================================
4. Standalone scale+bias epilogue after rocBLAS GEMM  [implement]
================================================================================

Sources (all verified):
- Grim's GEMM dispatch is plain rocBLAS top to bottom: roc_device.rs:1594
  `rocblas_gemm_strided_batched_ex` (split-k path), roc_device.rs:1711
  `rocblas_gemm_ex` / :1739 `rocblas_sgemm` (main path), matmul_batched:932.
  `grep hipblasLt hipblaslt` across the ROCm backend returns zero hits — there is
  no hipBLASLt reference anywhere, so "Rule 0 GEMM → rocBLAS/hipBLASLt" is not
  being declined: classic rocBLAS exposes NO epilogue-fusion API at all. A
  standalone post-GEMM kernel is the only structurally-possible option today,
  not a style choice.
- Precedent in-crate: `grim_broadcast_bias` at compute_kernels.rs:125 — extern
  "C" __global__, flat `batch * out_dim` indexing, launched via
  roc_device.rs:3524-3548 (`launch_broadcast`). That kernel is bias-only
  (writes `out[idx] = bias[col]`); it has no scale and no accumulation into
  existing GEMM output. The epilogue below is new logic, not an extension of an
  existing scale-aware kernel.

New kernel (compute_kernels.rs, next to grim_broadcast_bias):
```
// In-place scale+bias epilogue on a [batch, out_dim] GEMM output.
// out[i,j] = out[i,j] * a_scale[i] * b_scale[j] + bias[j]
extern "C" __global__ void grim_scale_bias_epilogue(
    float* out, const float* a_scale, const float* b_scale,
    const float* bias, int batch, int out_dim, int has_bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_dim;
    if (idx >= total) return;
    int i = idx / out_dim;   // token
    int j = idx % out_dim;   // output channel
    float s = a_scale[i] * b_scale[j];
    out[idx] = has_bias ? fmaf(out[idx], s, bias[j]) : out[idx] * s;
}
```
Block-dim: multiple of wave size (compute from batch*out_dim, clamp to 4 wavefronts
per block — mirror `choose_block_dim` reuse). One block launch, coalesced flat
indexing identical to `grim_broadcast_bias`.

Launcher (roc_device.rs):
- Add `launch_scale_bias_epilogue(&self, out, a_scale, b_scale, bias, batch,
  out_dim, has_bias) -> Result<hipStream_t>` mirroring the `launch_broadcast`
  pattern at roc_device.rs:3524-3548: resolve the `grim_scale_bias_epilogue`
  symbol from the kernel module, null-check args, launch on the active stream,
  return the stream. Same stream as the rocBLAS call that produced `out` so the
  epilogue is stream-ordered after the GEMM (rocm-hip-kernels integration
  checklist: bind library handle and custom kernels to the same stream).
- Call it in the W8A8 integer-GEMM path after the rocBLAS call when
  per-token `a_scale` and/or per-channel `b_scale` are present (the path
  `quantized_matmul` at roc_device.rs:2631 targets; the KQuant/MXFP4/FP8 branches
  already use fused kernels and don't need this).

Future fork point (NOTE, don't build):
- If grim later migrates the GEMM call itself to hipBLASLt
  (rust-ai-ml-inference-guide Action 9 prefers hipblaslt; it exposes native
  scale pointers via HIPBLASLT_MATMUL_DESC_*_SCALE_POINTER and algo heuristics),
  then a real choice opens between native epilogue fusion and this standalone
  kernel. Not relevant while the call is classic rocBLAS. Record as an ADR-able
  fork, no code.

Files touched:
- crates/grim-backend-rocm/src/kernels/compute_kernels.rs (kernel + source test).
- crates/grim-backend-rocm/src/device/roc_device.rs (launcher + W8A8 call site).

Acceptance:
- Parity test: build a random [batch,out_dim] GEMM output on CPU, apply
  a_scale/b_scale/bias in f32, run the kernel on the same data, assert close
  (1e-6 honestly — it is f32 fmaf).
- Sanity vs existing `grim_broadcast_bias`: with a_scale=b_scale=1 the two kernels
  agree on pure-bias output.
- `cargo test -p grim-backend-rocm --lib compute_kernels` green; no hipBLASLt
  symbol introduced.

================================================================================
5. Suggested commit split
===============================================================================

Independent, land separately (each compiles + tests green alone):

1. charon.rs fmt==7 (+ Q5_K/Q6_K)      -> hoisted batched Q4_K decode (item 1).
2. (evaluate-only)                     -> fp32 packed-atomic epilogue (item 2);
                                          record decision, no kernel change
                                          unless the device benchmark justifies it.
3. quantization.rs + accel_features.rs  -> Fp8NativeFormat split (item 3).
4. compute_kernels.rs + roc_device.rs  -> grim_scale_bias_epilogue (item 4).

===============================================================================
6. Verification commands (focused, not broad — grim-backend-correctness discipline)
===============================================================================

- Item 1: `cargo test -p grim-backend-rocm --lib charon`
          + on-device oracle-vs-fused parity test added in the same change
          + live model probe (GRIM_BACKEND set) output unchanged.
- Item 2: `cargo test -p grim-backend-rocm --lib charon` (no change expected);
          the deliverable is the written adopt/reject decision with the
          microbenchmark numbers.
- Item 3: `cargo test -p grim-backend-rocm --lib quantization`
          `cargo test -p grim-backend-rocm --lib accel_features`
- Item 4: `cargo test -p grim-backend-rocm --lib compute_kernels`

=================================================================================
7. Sources referenced
=================================================================================

- Old/reference: vllm csrc/rocm/qdq_4_rdna3.cuh (zero-FMA bf16-widen bit-trick;
  NOT copied — grim's nibble extraction is two-tier super-block keyed, and grim
  stays fp32; only the zero-FMA widening trick is borrowed, not the linear keying).
- `atomic_add_pk4_*`: generic packed-atomic family in LLM inference kernels
  (pack N narrow accumulation lanes into one wider RMW, unpack on read). Cited as
  the INSPIRATION for item 2 only; grim's fp32 output means the pattern is
  redesigned (128-bit CAS fp32x4 / 64-bit CAS fp32x2), not copied.
- grim-backend-rocm/src/kernels/charon.rs:556, 641-658 (fmt 7), 659-686 (fmt 8),
  778-784 (hot loop), 786-788 (output atomicAdd epilogue), 149/170/252/349/410/483/
  742 (all grouped variants share the same scalar atomicAdd epilogue),
  242/326/407 ("OCP" Microscaling naming).
- grim-quant/src/lib.rs:596-656 (dequant_q4k oracle, hoisted — the ground truth),
  1149/1215 ("OCP" naming).
- grim-backend-rocm/src/quantization.rs:7-22 (GcnArch), 98-112 (QuantMode),
  114-151 (QuantCapability), 154-216 (arch_capability / resolve_quant_mode).
- grim-backend-rocm/src/device/accel_features.rs:14-78 (mfma/wmma gates).
- grim-backend-rocm/src/device/roc_device.rs:1594/1711/1739 (rocBLAS dispatch),
  2631+ (quantized_matmul fused branches), 3524-3548 (launch_broadcast pattern).
- grim-backend-rocm/src/kernels/compute_kernels.rs:125-132 (grim_broadcast_bias).