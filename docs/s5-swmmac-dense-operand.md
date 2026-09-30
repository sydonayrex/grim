# S5 — Does SWMMAC permit a dense operand?

**Status: resolved. Outcome (a), with a caveat that re-cuts WS-E's claims.**

Answers the second of the two structural unknowns blocking GreyRaven (WS-E). The
first — whether the 2× is real on silicon — is step E7 and remains unmeasured.

## The question

`V_SWMMAC_F32_16X16X32_FP8_FP8` presents both operands expanded (A 16×32, B
32×16). The manual does not state whether one side may stay dense. This matters
because activations are dynamic and **cannot** be 2:4-pruned, so GreyRaven is
only useful if sparse-B / dense-A is legal.

## Why the plan's proposed probe cannot be performed

S5 in the plan proposed: *"a compile probe that attempts a SWMMAC with a 16×16
(dense) A operand, asserting the toolchain accepts it."*

**That probe is not performable as written.** Clang exposes exactly one SWMMAC
builtin signature per (accumulator type, tile shape, wave size). There is no
dense-A variant to attempt, so there is nothing to compile and nothing to assert.
The toolchain mirrors the manual's ambiguity rather than resolving it.

## What was established (toolchain surface, ROCm clang 22, gfx1201)

The GreyRaven instruction is real and supported:

```
__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32     <- exists
v_swmmac_f32_16x16x32_fp8_fp8                        <- exists (asm mnemonic)
```

The decisive observation is the **K dimension within the FP8 / f32-accumulate
slice** — the one GreyRaven actually uses:

| builtin | meaning | shape |
|---|---|---|
| `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | dense FP8 | 16×16×**16** |
| `swmmac_f32_16x16x32_fp8_fp8_w32` | sparse FP8 | 16×16×**32** |

There is **no** `swmmac_*_16x16x16_*` and **no** `wmma_f32_16x16x32_fp8_fp8_*`.
The K doubling from 16 to 32 **is** the 2:4 expansion: a dense 16×16 operand
becomes 16×32 when each group of 4 expands to 2 survivors plus position
metadata. So the sparse and dense forms are different *shapes*, not different
*modes*, and the toolchain offers no way to ask for one shape with a dense
operand.

**Scope of that claim, stated precisely — a broader version is false.** WMMA is
*not* universally K=16: `wmma_f32_16x16x32_f16` and `wmma_f32_16x16x32_bf16` both
exist, and `wmma_i32_16x16x32_iu4_w32_gfx12` is a K=32 WMMA with 8-bit-ish
elements (int4, packed 8-per-register). Those are 16-bit-element and
int4/i32-accumulate paths. What does *not* exist anywhere in the toolchain is an
**f32-accumulate, 8-bit-element WMMA at K=32**, or **any SWMMAC at K=16**. The
asymmetry that matters is therefore specific to GreyRaven's data type, and an
earlier draft of this document over-claimed it as global.

Note the SWMMAC builtins carry **no `_gfx12` suffix**, unlike the WMMA family.
Gating is by subtarget feature rather than by a suffixed builtin name.

## The resolution: dense A is legal, via zero-fill — at a cost

A dense A is achievable by **zero-filling into the expanded 16×32 form**: place
the dense 16×16 activations in their correct slots and write zero into the
pruned slots. Nothing in the ISA requires the A-side values to be genuinely
sparse; "sparsity" here is a register-layout contract, and zero is a legal value
in a pruned slot. The product then equals dense-A × sparse-B.

This makes outcome (a) — sparse-B / dense-A legal — but the caveat is the whole
point, and it is not a small one:

- **The 2× MAC rate is available.** It is an SWMMAC; the rate applies.
- **The B-side (weight) saving is real.** 2:4 on weights, which is where the
  parameter bulk lives. This is the claim GreyRaven actually rests on.
- **The A-side saving is NOT available.** A occupies the expanded footprint —
  2× the dense 16×16 — so activations cost exactly what a sparse operand costs.
  Any design that claims an activation-side memory win is wrong.
- **A zero-fill cost lands on the critical path**, proportional to K per tile.
  Step E7's timing must include it, or GreyRaven will look faster than it is for
  exactly the reason it is not.

## Consequence for WS-E

**Step E8 is no longer blocked**, so WS-E is not killed on S5 grounds. But its
scope must be re-cut to claim **weight-side sparsity only**. The 4.75 bpw figure
from the host-side sparsifier (commit `0de43b90`) is a *weight* figure and
remains valid; no equivalent figure exists for activations.

## Limits of this investigation — read before relying on it

Stated plainly, because the confident-sounding conclusion above outruns what was
actually verified:

1. **RESOLVED — the builtin signature is now determined.** The earlier searches
   used inline-asm forms and did not converge. Letting the type checker drive it
   converges immediately, because the diagnostics name the expected type. For
   ROCm clang 22.0.0git, verified by compiling and reading the emitted IR:

   ```c
   // v8f32 __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
   //     v2i32 d, v4i32 a, v8f32 b, u32 c);
   typedef int  v2i __attribute__((vector_size( 8)));  // d
   typedef int  v4i __attribute__((vector_size(16)));  // a
   typedef float v8f __attribute__((vector_size(32))); // b, and the return
   ```

   The `d` operand is `__vector_size__(2 * sizeof(int)) int`, `a` is
   `__vector_size__(4 * sizeof(int)) int`, `b` is
   `__vector_size__(8 * sizeof(float)) float`, and `c` is a plain `int` — it is
   *not* a vector, which the "all four are vectors" assumption in the original
   search space is what made the search diverge. `b` accepts either a v8 of int
   or a v8 of float, so the FP8 payload is passed as raw bits and the element
   type is not the discriminator.

   The builtin requires target features `gfx12-insts` and `wavefrontsize32`, and
   lowers to `llvm.amdgcn.swmmac` on **both** gfx1200 and gfx1201 (verified by
   `-S -emit-llvm`). The 2× MAC-rate claim is still a vendor table entry until
   E7 measures it; that part of this investigation is unchanged.

   **RESOLVED, and it inverts a design assumption.** The asymmetry above is not an
oddity once the intrinsic's own definition is read. ROCm LLVM ships the
tablegen class in `llvm/IR/IntrinsicsAMDGPU.td`:

```
def int_amdgcn_swmmac_f32_16x16x32_fp8_fp8
  : AMDGPUSWmmacIntrinsicIdx<llvm_anyint_ty, llvm_anyint_ty,
                             llvm_anyfloat_ty, llvm_anyint_ty>;

class AMDGPUSWmmacIntrinsicIdx<LLVMType A, LLVMType B, LLVMType CD, LLVMType Index>
  : Intrinsic<[CD], [ A /*%A*/, B /*%B*/,
                      LLVMMatchType<0> /*%C*/, Index /*Sparsity index for A*/ ], ...>;
```

Aligning the IR operand list with the measured clang signature gives a
self-consistent reading, and the byte sizes cross-check it:

| param | type  | bytes/lane | total  | meaning                                       |
|-------|-------|-----------|--------|-----------------------------------------------|
| arg0  | v2i32 | 8         | 256 B  | **A** — 16x32 FP8, *sparse* (512 B, half stored) |
| arg1  | v4i32 | 16        | 512 B  | **B** — 32x16 FP8, dense                       |
| arg2  | v8f32 | 32        | 1024 B | **C/D** — 16x16 f32 accumulator (256/32 = 8)   |
| arg3  | u32   | —         | —      | **sparsity index for A**                       |

A 16x32 FP8 operand is 512 B; stored 2:4-sparse that is 256 B = v2i. B at
32x16 is 512 B = v4i. A 16x16 f32 tile is 1024 B = v8f, which is why the
*return* is 8 floats per lane and arg0 is not the accumulator at all.

**The consequence: this intrinsic takes its sparse operand on A, and the
sparsity index is explicitly "for A".** GreyRaven is specified the other way
round — sparse *weights*, dense *activations* — and the plan says so itself
(`activations are dynamic and cannot be 2:4-pruned`). So the one available
sparse FP8 intrinsic has its sparsity on the wrong operand for this format.

This inverts the earlier conclusion in this document, which reasoned that
sparse-B/dense-A was legal via zero-fill into an expanded A. That reasoning
assumed the expansion happened on the sparse side. On this intrinsic the
sparse side is A, so zero-filling *B* would buy nothing and zero-filling *A*
would mean pruning the activations, which the format forbids.

This is inference from the intrinsic definition plus the measured signature,
not from execution -- E6 still has to run it. But it is strong enough to
change the plan before E6 is written rather than after, because the work E6
would otherwise build is a kernel for the wrong sparsity placement.

Two ways forward, both needing a decision rather than more investigation:

  1. **Transpose the problem.** Feed the *weights* as A and *activations* as B,
     computing C^T = B^T x A^T. That puts the sparse operand on A as the
     hardware wants, at the cost of a transposed output layout and an
     activation operand that is the *narrow* side of the tile.
  2. **Abandon the sparse FP8 path** and reconsider GreyRaven as a
     weight-pruned format that reconstructs dense B and uses dense WMMA -- which
     is at 8 bpw for FP8, worse than MXFP4's 4.25, and would need the
     activation side to carry the density instead. On the evidence so far this
     is the weaker option, and E9's matched-tolerance comparison is what should
     decide it.

**One asymmetry is not yet explained and E6 must settle it on hardware:** `d`
   is a 2×i32 input but the result is 8×f32. A 16×16 f32 accumulator over a
   32-lane wave is 8 floats per lane, so the *result* width is the expected one
   and the *`d`* width is not. Either `d` is not the accumulator seed, or the
   intrinsic has a quirk where the accumulator-in operand is narrower than the
   accumulator-out. The type is known; the role of `d` is not, and guessing it
   would produce a kernel that compiles and returns plausible numbers.
2. **The zero-fill argument is still reasoned, not compiled.** The signature is
   now known, so this is finally a writable experiment rather than a shape
   argument, but it has not been run: no probe yet assembles a
   dense-A-in-expanded-form SWMMAC and checks the result against a dense-A
   dense-B reference. **E6 and E8 should treat this as the hypothesis to test
   first.** It is now a one-kernel experiment with a known-good signature, which
   is a materially different position from where this document was written.
3. **No GPU was involved.** Compiler only: `clang++ --target=amdgcn-amd-amdhsa
   -mcpu=gfx1201` and `llvm-mc`. Nothing was executed on hardware, and E7's 2×
   claim remains a vendor table entry until measured.

## Reproduce

```bash
C=/opt/rocm/llvm/bin/clang

# the two load-bearing negatives. Both must print nothing.
strings $C | grep -oE '__builtin_amdgcn_swmmac_[a-z0-9_]*16x16x16[a-z0-9_]*' | sort -u
strings $C | grep -oE '__builtin_amdgcn_wmma_f32_16x16x32_fp8_fp8[a-z0-9_]*' | sort -u

# and the two positives that establish the 16 <-> 32 expansion. Each prints two
# names, one per wave size; gfx1201 is wave32, so the _w32 form is the one that
# matters for GreyRaven.
strings $C | grep -oE '__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8[a-z0-9_]*' | sort -u
strings $C | grep -oE '__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8[a-z0-9_]*' | sort -u

# the broader K survey, which shows why the claim must be scoped to FP8
strings $C | grep -oE '__builtin_amdgcn_wmma_[a-z0-9_]*16x16x32[a-z0-9_]*' | sort -u
```

**Wave size.** Both SWMMAC forms exist as `_w32` and `_w64`. gfx1201 is a
wave32 part, so `_w32` is the relevant one; the `_w64` presence is noted only so
nobody reads its absence from a grep as a gap.
