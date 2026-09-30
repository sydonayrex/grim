# GreyRaven: weight-sparse on the A operand

Status: **validated on hardware.** `grey_raven_sparse_gemm_matches_dense_reference`
passes with 0 of 256 mismatches, against a reference transcribed from the RDNA4 ISA
(see `old/amd-isa/rdna4-instruction-set-architecture.pdf`, sections 7.12, 7.12.2,
7.12.3). Everything below is measured, not inferred.

## Why the weights go on A

The only sparse FP8 instruction in the ISA surface takes its sparsity on **A**:
`V_SWMMAC_F32_16X16X32_FP8_FP8`, with `D0.f32(16x16) = S0.fp8(16x16) *
S1.fp8(32x16, index set from S2) + D0.f32(16x16)`. B must be dense. There is no
sparse-B form, and the `ABIdx` family has no FP8 variant at all.

So a sparse-weight format has exactly one place to put the weights: MMA **A**.

The operand order follows from the instruction's own definition, not from a
prototype: assemble it and read the VGPR operand widths, which admit only
`(v2i32, v4i32, v8i32, i32)`. A is 2 VGPRs, B is 4, and there is no permutation
that type-checks. The earlier hand-written prototype in `s5-swmmac-dense-operand.md`
had the same answer but was never checked; the check is now mechanical.

## The fragment, as measured

| | |
|---|---|
| **A** | *packed* 16x16 (M x K/2), 8-bit, wave32: `lane = {col[3], row[3:0]}`, `vgpr = col[2]`, `startPosn = col[1:0]`. Expands to 16x32. |
| **B** | 32x16, **column-major**: `lane = (k>>4)*16 + n`, byte within lane = `k & 15`. |
| **C/D** | 16x16, 32-bit: `lane = {row[3], col[3:0]}`, `vgpr = row[2:0]`. Read *and* accumulated, so it is both C and D. |
| **index** | per lane. Group `c` (dense `k` 4c..4c+3) uses packed cols `2c`, `2c+1`; `idx0` at bits `[4c+1:4c]`, `idx1` at `[4c+3:4c+2]`, with **`Idx0 < Idx1`**. The word for group `c`, row `r` lives in lane `(c>>2)*16 + r` at bit offset `4*(c & 3)`. |

## What this means for the format

- **No change is needed.** GreyRaven's 3 metadata bits per group encode
  `(Idx0, Idx1)` directly.
- **All six 2-of-4 pairs are expressible.** `idx0` and `idx1` are both free 2-bit
  positions, so the reachable set is the full `C(4,2)`. There is no anchor at
  slot 0. An anchored pruner was written and reverted (`fe762ae6`) because it
  halved the pattern space for a constraint that does not exist.
- **K = 32 per row, one instruction, 16 survivors** — which is the 4.75 bpw the
  format has claimed throughout: 2 E4M3 survivors (16 bits) + 3 metadata bits per
  group of 4.
- Decode needs no transpose. At `M = 1` the output is `1xN`, and `N x 1` is also
  N contiguous floats, so putting W on A costs nothing in tile waste at decode.
  Prefill, where `M` is large, does pay for the tile reuse; that has not been
  measured.

## Two traps, both of which bit during this work

**`idxFirstBit = col[3:2] * 4` is a two-bit field.** `col` is `4c`, so `col[3:2]`
is `(4c >> 2) & 3 = c & 3` and the offsets are only ever 0, 4, 8, 12. Reading it
as `4 * c` is the natural mistake and produces the `S = 0.5` VGPR discrepancy in
Table 43 — four groups, eight 2-bit values, sixteen bits per lane.

**`idx1` can land on col+1 or col+2, not only col+3.** A reference that consults
only `idx0` for the first three columns drops survivors and understates the
expected value, which makes a *correct* product look wrong.

## Superseded reasoning, kept so it is not re-derived

Three claims from the reverse-engineering sequence were wrong and are withdrawn:

- "the effective contraction is 16x16x16" — A is packed 16x16 *expanding to*
  16x32; `k = 0..31` is all real.
- "B is lane = column, byte = k within a 16-half" — B is column-major.
- "the index is a scalar" — it is per lane.

The first came from packing B row-major, which left the real `k = 16..31` half
empty, so the product matched a reference built to the same wrong shape and the
test passed for the wrong reason. The third came from driving the index with one
repeated field: a field of 0 makes `idx0 = idx1 = 0`, violating `Idx0 < Idx1` and
degenerating to a single survivor at position 0, which reads exactly like a
hardware anchor.
