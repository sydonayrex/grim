# GreyRaven, reformulated: weight-sparse on the A operand

**Status: design decision, taken 2026-09-29. Supersedes the operand placement in
`docs/s5-swmmac-dense-operand.md` and in `plans/PLAN-corvid-precision.md` §WS-E.**

## The constraint

The only sparse FP8 instruction in the ISA surface takes its sparsity on **A**,
and names the fourth operand "Sparsity index for A". This is not a wart of one
intrinsic; it is the whole family:

- `AMDGPUSWmmacIntrinsicIdx` — the `16x16x32_fp8_fp8` family. `%Sparsity index for A`.
- `AMDGPUSWmmacIntrinsicABIdx` — the `16x16x64_f16/bf16` family. Also
  `%Sparsity index for A`, plus per-operand sign modifiers and A/B reuse flags.

There is no sparse-B form, and the `ABIdx` family has no FP8 variant at all. So
"put the sparsity on the weights, as the hardware wants it" is not available;
the only question is which logical operand becomes A.

## The reformulation

GreyRaven computes the **transpose** of the usual linear:

```
natural      MMA_A = Act   (M x K)   MMA_B = W^T (K x N)   C = M x N
reformulated MMA_A = W     (N x K)   MMA_B = Act^T(K x M) C = N x M
```

Weights become the sparse A operand, which is what the instruction wants.
`W` is still 2:4-pruned along K, and K is the MMA's 32-deep axis, so the
existing prune is already laid out correctly — **the format does not change,
only which operand the kernel feeds it to.**

## Why the transpose is free where GreyRaven matters

GreyRaven exists for **decode**, which is M = 1. At M = 1 the transpose has no
cost, in either place it could have one:

**Output layout.** Natural C is 1×N; reformulated C is N×1. Both are *N
contiguous floats*. A single row and a single column are the same memory. There
is nothing to transpose, so the epilogue needs no transpose and no scratch.

**Tile occupancy.** WMMA wastes 15/16 of a tile at M = 1 wherever the narrow
dimension lands. Natural puts the narrow axis in A (15/16 of the A tile idle);
reformulated puts it in B (15/16 of the B tile idle). The waste is the same size
in both — it is inherent to WMMA at M = 1, which is precisely why this codebase
has an entire `dot_gemv` family of non-WMMA decode kernels. Transposing does not
create the waste and does not remove it.

**The value proposition.** GreyRaven's reason to exist at decode is halving the
weight bytes read, because decode is weight-bandwidth-bound. W is 2:4-pruned in
both orientations, so that saving is untouched.

So at M = 1 the reformulation costs **zero** and changes nothing about the
density claim.

## What it costs, and where that lands

Only prefill (M > 1), where C is genuinely N×M and needs a transposing epilogue
or a separate transpose kernel. That is the right place to pay it:

- Prefill is compute-bound. Weight reads amortise over M tokens, so halving
  them buys proportionally less the wider the tile.
- 2:4 weight sparsity is at its weakest in exactly this regime.

So the shape where the rewrite costs something is the shape where the format was
already worth least. A prefill implementation can also simply be omitted until
E9 says the format is worth keeping at all — E9 is a matched-tolerance
comparison at decode shapes, which is where the decision lives.

## What stays true, and what must be re-argued

Unchanged: the 4.75 bpw storage cost (2 survivors × 8 bits + 3 metadata bits per
group of 4); the E1/E2 Fisher-guided prune; the E4/E5 pack/dequant semantics;
the requirement that a dense FP8 model of the same values carry identical codes.

Must be re-argued: every place the plan described GreyRaven as "sparse weights,
dense activations" without saying which operand that meant. The plan's own note
that activations cannot be 2:4-pruned is *why* this reformulation is necessary,
not an obstacle to it.

## Open items this does not settle

- **The sparsity index encoding.** The `u32` per-lane index is hardware state
  built at load time, not stored, so it does not change the 4.75 bpw *storage*
  claim — but its bit cost is unknown and it competes for VGPRs alongside the
  fragments. That is a register-pressure question E6/E7 must measure, not a
  density one.
- **The index-to-pattern mapping.** Whether the hardware expects one bit per
  4-element group, a compacted code, or a per-lane layout is still undetermined.
  It has to match the 3-bit metadata the packer emits, and that correspondence is
  E6's first job.
- **E7's 2× rate** remains a vendor table entry until measured, and it now
  applies to the reformulated kernel rather than any kernel written so far.
