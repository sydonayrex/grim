# Spec: `qkv_attention.rs` Phase-2 — Bandwidth & Occupancy Optimization

**Status:** Draft — pending review
**Component:** `grim-backend-rocm` / `qkv_attention.rs`
**Depends on:** Phase-1 (correctness-first fused QKV attention kernel, current state)
**Skills:** `rocm-hip-kernels`, `rust-gpu-discipline`, `rust-gpu-parallelism`
**Related work reviewed:** FlashAttention-4 (Zadouri/Hoehnerbach et al., arXiv:2603.05451),
Modal "Making FlashAttention-4 faster for inference" (modal.com/blog/flash-attention-4-faster)

---

## 1. Problem Statement

Phase-1 `qkv_attention.rs` (`grim_qkv_attention`, `grim_qkv_attention_paged`,
`grim_tree_attention`) is **numerically correct** but does no memory-hierarchy
optimization: every K/V element is re-read from global memory on every query's
pass, Q is re-read from global memory on every inner-loop iteration, and
parallelism is capped at 4–8 wavefronts within a single block regardless of
how much of the GPU is idle. This is architecturally closer to naive
attention than to FlashAttention.

Two independent external sources — the FA-4 paper's own roofline analysis and
Modal's field report on adapting FA-4 for inference — converge on the same
two root causes, in the same priority order, for a kernel in this state:

1. **Memory traffic, not compute, is the bottleneck.** The FA-4 paper's
   roofline analysis (§3.1.1, §3.2.1) shows shared-memory traffic exceeding
   MMA compute time by 25–60% even in an already-tiled kernel; our kernel
   has *zero* tiling, so the gap versus a bandwidth-optimal implementation is
   categorically worse.
2. **Query-only parallelism starves decode.** Modal (PR 1940) found FA4 was
   *slower than FA2* on B200 for small-batch decode because grid parallelism
   was query-dimension-only, leaving up to 75% of SMs idle. Our grid
   (`seq_len, num_heads, 1`) has the identical shape and the identical
   failure mode — for `seq_len_q = 1` decode, only `num_heads` blocks launch
   total, each internally capped at 4–8 wavefronts of KV parallelism.

This spec defines an ordered set of changes to close both gaps, each
independently implementable and independently verifiable against the
Phase-1 kernel's output.

## 2. Non-Goals (deferred to later phases)

- **Numeric precision changes** (FP16/BF16/FP8 K/V cache). Real win per
  Modal PR 2109, but orthogonal to kernel structure and requires its own
  accuracy validation pass. Not in scope here.
- **Tensor-core (MFMA) dot products.** QK^T/PV remain scalar FMA in this
  phase. Whether to route through MFMA (and via `wmma_gemm.rs` or a new
  attention-specific MFMA path) is an open architectural question — see
  §7 Open Questions — not a decision this spec makes.
- **Software-emulated exponential** (FA-4's polynomial-on-FMA trick).
  Requires its own roofline analysis of RDNA's transcendental-vs-FMA
  throughput ratio, which does not yet exist for our target archs
  (gfx1036/gfx1100/gfx1200). Do not port FA-4's technique by analogy without
  that analysis — the underlying hardware asymmetry it exploits is
  Blackwell-specific and unconfirmed for RDNA.
- **`flash_attn.rs`.** A separate, currently non-tiled and (per prior review)
  possibly correctness-broken kernel file. Out of scope for this spec;
  tracked separately pending confirmation of whether it is dead code or
  live-and-broken.

## 3. Ordered Work Items

Each step below must pass the Step 0 regression harness before the next
step begins. This ordering is deliberate: cheap/zero-risk changes first,
the highest-leverage structural change third, and the highest-complexity
change last.

### Step 0 — Baseline & Regression Harness (prerequisite, no kernel changes)

**What:** Establish (or confirm existing) reference-output tests comparing
kernel output against a naive CPU/reference softmax(QK^T)V implementation,
covering:
- Non-causal and causal masking
- GQA (`num_heads != num_kv_heads`) and MQA (`num_kv_heads = 1`)
- `grim_qkv_attention`, `grim_qkv_attention_paged`, `grim_tree_attention`
  paths independently
- `head_dim` values spanning the wave32/wave64 boundary and the 256 cap
- Edge cases: `kv_seq_len = 1`, `page_size` not dividing `kv_seq_len` evenly,
  `cache_offset > 0`

**Why:** Every subsequent step is a claim of "faster, same output." Without
this harness in place first, that claim is unverifiable and steps 3–4 in
particular are high-risk to land silently-wrong (per fail-loud/single-source-
of-truth: an unverified performance claim is not different in kind from an
unverified correctness claim).

**Acceptance:** Harness exists, passes against current Phase-1 kernel,
committed before any Step 1+ work begins.

---

### Step 1 — Eliminate Redundant Q Reload

**What:** `q[q_offset + dim]` is currently read from global memory inside
the `j` loop despite being loop-invariant. Load the thread's Q elements
into registers once, before the loop, in all three kernel bodies.

**Why:** Zero algorithmic change, pure redundant-load elimination. Cheapest
possible first commit; establishes the pattern of "verify against Step 0"
before anything riskier.

**Acceptance:** Bit-identical output vs. Step 0 baseline. Measurable
reduction in global memory transactions (verify via `rocprof-compute`
per the `rocm-hip-kernels` checklist item — "Validated with rocprof-compute
... before claiming 'fast'").

---

### Step 2 — Fix Wasted `expf` in Paged/Tree Kernels

**What:** `grim_qkv_attention_paged` and `grim_tree_attention` currently
compute `w = expf(score - running_max)` unconditionally, then discard and
recompute `scale` conditionally when `score > running_max`. Align both to
the cleaner branch structure already used correctly in `grim_qkv_attention`
(compute `scale_old`/`scale_new` conditionally, never compute a value that
gets thrown away).

**Why:** Removes a wasted transcendental-unit call on the common branch in
two of the three kernels. Isolated, mechanical, easy to verify.

**Acceptance:** Bit-identical output vs. Step 0 baseline (this is a
dead-code-elimination change, not a numerics change — output must not
shift at all).

---

### Step 3 — LDS Tiling of K/V (primary structural change)

**What:** Restructure the inner loop from "one global load per (thread, j)"
to cooperative block-level tiling:

- Choose a KV tile size `TILE_KV` (candidate: 32 or 64 positions) sized to
  fit LDS alongside the existing `s_max[8]` / `s_sum[8]` / `s_acc[8][256]`
  arrays, within `hipDeviceProp.sharedMemPerBlock` for the target archs.
- Each wavefront cooperatively loads one K-tile and one V-tile into LDS
  (coalesced load — consecutive lanes read consecutive addresses, per
  `rocm-hip-kernels` LDS guidance), `__syncthreads()`, then all threads in
  that block iterate the online-softmax update over the tile from LDS
  rather than HBM.
- Double-buffer the LDS tile (ping-pong) to overlap next-tile load with
  current-tile compute, per house convention.

**Why:** This is the change the FA-4 paper's roofline analysis identifies
as necessary at any generation of hardware, and the one most clearly
absent from Phase-1: right now every query re-reads the *entire* assigned
K/V range from global memory once per query position, i.e. O(seq_len ×
kv_seq_len) HBM traffic where a tiled kernel achieves O(seq_len +
kv_seq_len). This is expected to be the largest single improvement in this
spec.

**Acceptance:**
- Bit-identical (or within acceptable float-accumulation-order tolerance —
  define explicit epsilon, do not silently allow drift) output vs. Step 0
  baseline across the full harness, including the causal-masking and
  GQA/MQA cases.
- Measured reduction in global memory traffic proportional to tile size,
  confirmed via `rocprof-compute`.
- LDS usage stays within `sharedMemPerBlock` for all three target archs
  (gfx1036, gfx1100/1101/1102/1103, gfx1200/1201) — verify against queried
  device properties, not hardcoded assumptions.

**Risk:** Highest-complexity change in this phase after Step 4. Changes to
the causal masking logic (`hi`, `range_len`) and the paged/tree KV-address
resolution (`BlockTableEntry` lookup, `is_ancestor` walk) must be
re-derived against tile boundaries rather than per-element `j`, since tile
loads cross page/ancestor boundaries that the current per-element logic
doesn't need to reason about.

---

### Step 4 — Grid-Level KV Split for Decode

**What:** Promote KV parallelism from "4–8 wavefronts within one block" to
grid level for the decode-shaped case (`seq_len_q` small relative to
`kv_seq_len`):

- New launch path: multiple blocks per query, each owning a KV shard,
  writing partial `(running_max, running_sum, out_acc)` to a scratch global
  buffer instead of resolving to final output directly.
- New combine kernel: reduces per-block partials into final output, using
  the same running-max/running-sum merge algebra already implemented for
  the wave-0 LDS merge in Step 3 — promoted one level, not reinvented.
- Split-count heuristic: start with a simple function of `kv_seq_len` and
  available CU count (empirical, not analytically derived — consistent
  with "simplest thing that works" per project convention); refine only
  after real `rocprof-compute` numbers are in hand.

**Why:** Modal's PR 1940 finding applies directly: for decode-shaped
workloads (`seq_len_q` = 1 or a handful of speculative tokens), the current
grid shape (`seq_len, num_heads, 1`) launches far fewer blocks than there
are CUs to fill, and the intra-block wavefront split (Step 3's target)
cannot compensate — 4–8 wavefronts is a rounding error against a whole
GPU's CU count for `seq_len_q = 1`.

**Acceptance:**
- New combine-kernel path produces output within the same epsilon
  tolerance as Step 3's single-block path, on identical inputs, across
  the full harness.
- Demonstrated occupancy improvement (active wavefronts / max wavefronts
  per CU, via `rocprof-compute`) for `seq_len_q` = 1 and small-batch
  shapes specifically — this is the case Step 3 alone does not fix.
- Split heuristic does not regress prefill-shaped workloads (`seq_len_q`
  large) relative to Step 3's baseline — verify both regimes, not just
  decode.

**Risk:** Largest scope change in this phase — new kernel, new launch
function, new scratch-buffer lifetime to manage (device-resident, per
`rust-gpu-discipline`; no host readback mid-pipeline). Recommend this step
get its own focused review pass before merge, separate from Steps 1–3.

---

### Step 5 — Cleanup (ride-along, no independent acceptance gate)

Since every kernel body is already being touched in Steps 1–4, fold in:

- Remove unused `thread_active` (computed, never read, in all three
  kernels).
- Remove the shadowed top-level `const int d = lane_id;` declaration,
  which is immediately re-declared inside every loop as
  `int d = lane_id + chunk * wave_size` — the outer one is dead.
- Add an explicit cross-reference comment linking the HIP-source
  `BlockTableEntry` struct (line ~195 in current file) to the
  `#[repr(C)]` Rust `BlockTableEntry` (line ~511) — a silent layout
  mismatch between these two is a correctness bug with no compiler
  signal, which is exactly the class of risk the project's single-
  source-of-truth principle exists to prevent. If feasible, add a
  `static_assert(sizeof(BlockTableEntry) == N, ...)` on the HIP side
  keyed to the Rust struct's known size.

## 4. Verification Strategy (applies across all steps)

Per the `rocm-hip-kernels` grim integration checklist, before any step in
this spec is described as "done":

- [ ] Block size remains a multiple of 64 (Wave64 mandate) across all new
      launch configurations introduced in Steps 3–4.
- [ ] Coalesced global loads confirmed for the new LDS tile-load code in
      Step 3 (not just assumed from access-pattern inspection).
- [ ] LDS usage validated against queried `sharedMemPerBlock`, not
      hardcoded, for all three target archs.
- [ ] No host readback introduced anywhere in the Step 4 scratch-buffer
      combine path.
- [ ] Each step validated with `rocprof-compute` occupancy + stall
      counters before being described as a performance win — a claimed
      speedup without a profiler number attached is not accepted per
      house convention.
- [ ] Full regression harness (Step 0) green after every step, not just
      at the end.

## 5. Explicit Traceability to Source Findings

| Change | Source finding | Where |
|---|---|---|
| Step 1 (Q reload) | Redundant global load, no external source needed — direct code inspection | `qkv_attention.rs` inner loop |
| Step 2 (wasted `expf`) | Non-matmul/transcendental-unit bottleneck | FA-4 paper §3.1.3 (exponential throughput as bottleneck, general principle; specific bug is local) |
| Step 3 (LDS tiling) | Shared-memory traffic exceeds MMA compute by 25–60% (fwd), ~30% (bwd) in an *already-tiled* kernel | FA-4 paper §3.1.1 Table 1, §3.2.1 Table 3 |
| Step 4 (grid-level KV split) | FA4 "generally slower than FA2 on B200s" pre-fix; up to 75% of SMs idle under query-only parallelism | Modal blog, PR 1940 section |

## 6. Explicit Non-Transfers (things NOT ported from FA-4, and why)

To avoid over-applying Blackwell-specific findings to RDNA hardware without
justification:

- **2-CTA MMA mode / tensor memory (TMEM) tricks** (FA-4 §3.2.3): Blackwell-
  specific hardware feature with no RDNA analog. Not applicable.
- **Software-emulated exponential via FMA polynomial** (FA-4 §3.1.3): valid
  only because Blackwell's MUFU throughput (16 ops/clock/SM) is dramatically
  outpaced by its tensor cores (8192 ops/clock/SM) — an asymmetry specific
  to that generation's scaling. RDNA's FMA-vs-transcendental ratio has not
  been profiled for our target archs; porting this technique by analogy
  without that data would be exactly the "reframe to make it seem safe"
  pattern this project's engineering culture rejects. Deferred, see §2.
- **CuTe-DSL / Python-embedded kernel authoring**: a tooling/compile-time
  choice orthogonal to kernel algorithm; not relevant to a Rust/hipRTC
  JIT pipeline.

## 7. Open Questions (require a decision before or during implementation)

1. Should QK^T/PV route through MFMA intrinsics (per `rocm-hip-kernels`
   guidance) as part of or after this phase? Current scope keeps them
   scalar; this is a bigger structural change than anything in this spec
   and needs its own decision point, not a default assumption.
2. What is the accuracy epsilon tolerance for Steps 3–4 given
   floating-point non-associativity from tile-order and split-order changes
   (the same effect Modal notes for split-KV: "summing within a split, then
   across them, gives different results from summing across the flat
   sequence")? Needs an explicit number before Step 3 acceptance criteria
   can be finalized, not left implicit.
3. Is `flash_attn.rs` live or dead code? Blocks nothing in this spec
   directly but should be resolved before or in parallel, since it's a
   correctness question (P0-class) rather than a performance one.

## 8. Sequencing Summary

```
Step 0 (harness) → Step 1 (Q reload) → Step 2 (expf fix) →
Step 3 (LDS tiling) → Step 4 (grid KV split) → Step 5 (cleanup, ride-along)
```

Steps 1–2 may land as a single small PR. Step 3 and Step 4 should each be
their own PR with independent review, given their risk profile per §3.
