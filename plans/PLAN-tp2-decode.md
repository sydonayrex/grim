# PLAN: TP=2 decode across both RDNA4 cards (REV 2 — re-derived from the measured decomposition)

REV 2 supersedes the original. REV 1 was written before the per-kernel
decomposition populated (the GRIM_KERNEL_TIMES cached path wrote to a map
nothing read — fixed in ae75e0c6) and carried three errors this revision
corrects: the 8/56 layer mix (actual: **16 full-attention / 48 KDA**,
`layer_types` = 3 linear + 1 full repeating), a residency budget that
double-counted packed+lanes past 15.92 GiB, and a bandwidth-only wall
model that the measurement has falsified. Baselines updated to the
32/32-split measurements.

## Measured ground truth (2026-10-31, qwen35_production_27b, GRIM_SPLIT_LAYERS=32,32)

- Wall: 92.3 ms, warm == cold (+6 µs) — managed memory gone, not paging.
- Kernel busy (event-timed eager walk, cold-L2): **27.6 ms**.
  Roofline floor: 26.8 ms. Busy ≈ roofline — the kernels are near their
  own ceilings.
- **Inter-kernel gap ≈ 65 ms = 70% of the wall** (1544 nodes / 2 segments
  ≈ 42 µs per node). This is the first-order cost.
- Route: 812 launches/step. Busy leaders: iq4xs GEMV 6.4 ms/63,
  rms_norm 5.3 ms/80 (66 µs each — ~13x too high for w5120, unexplained),
  W4A4 v5 3.8 ms/59, Q5_K 2.6 ms/43, iq3s grouped 2.3 ms/4 (573 µs each).
- Cards: GPU0 = RX 9070 XT (Navi48), GPU1 = RX 9060 XT (Navi44) —
  different SKUs; any split is weighted by P0's measured per-card BW.

## Consequence for TP (the viability statement)

TP=2 halves the busy fraction at best: overlapped busy floor ≈
max-per-card(half-busy) ~ 14-18 ms + sync. It does nothing about the gap —
worse, it grows it: ~1544 → ~3100 nodes and ~130-190 cross-graph
rendezvous/token in a system whose dominant cost is already per-node gap.

**Therefore the ≤ 45 ms target is NOT viable from TP alone and REV 2
withdraws it.** Viable sequencing:

- **Stage A (precondition, independent of TP):** attack the 65 ms gap —
  node/segment structure, the 325 quantize-prologue nodes, rms_norm 66 µs,
  iq3s 573 µs. Every ms removed here is a ms TP also keeps.
  - **Stage A Status: COMPLETE (2026-11-01, f063327a).**
    - Step 1 (Residual 1 fusion + fast rms_norm + iq3s_decode): nodes 1543 -> 1479, wall 75.3ms -> 69.9ms (-5.4ms).
    - Step 2 (Residual 2 fusion into next-layer norm): nodes 1479 -> 1416, wall 69.9ms -> 69.9ms, rms launches 161 -> 34.
    - Step 3 (Sibling quantize dedup q8_1 & u4): nodes 1416 -> 1327, wall 69.9ms -> 69.6ms, launches 1487 -> 1398 (q8_1 285->217, u4 103->82).
    - Overall Stage A: cold wall 75.3ms -> 69.6ms (clock-pinned 'high'), busy 66.9ms -> 59.6ms, total launches 1614 -> 1398 (-216 launches), bit-exact 0.000e0 across all 248,320 logits, 512-token |ΔNLL| = 0.000000 nats (100.00% top-1 agreement).
- **Stage B:** re-derive the TP target from the post-Stage-A wall:
  TP ceiling ≈ (post-A busy)/2-overlapped + sync, and TP is worth building
  only if that beats the post-A serial wall by the measured sync cost.
  Predeclared criterion, no target number until A lands.
- **What TP is unconditionally good for** (independent of the wall): the
  u4-lane mirror disappears (each card converts only its shards) and the
  9060 XT's idle cycles get used. These hold even if the wall gain is small.

## Model geometry (verified against config.json)

- 64 layers: **16 full-attention, 48 KDA** (interval 4). hidden 5120,
  INTER 17408, vocab 248320.
- Attention: 24 q × head_dim 256, 4 kv heads, attn_output_gate=true —
  the [q|gate] channel pairing means each gate head MUST split with its q
  head (even 12q/2kv per card preserves this; it is a hard constraint on
  any weighted split: q-head splits must be gate-paired).
- KDA: 16 key heads, 48 value heads × 128, conv kernel 4, conv state is
  [q|k|v] concatenated over 10240 channels — head-split slices CHANNELS of
  that layout; the slice map (which channel ranges belong to which head)
  must be written down in P1, not assumed.
- GQA group: 6 q per kv (24/4). Per card at 12/12+2/2: 6 q per kv intact.
- KDA state: 48 × 128² per head — per-head, no cross-head recurrence terms;
  splits exactly.

## Residency (corrected)

The u4 lanes REPLACE the packed blob at decode — one representation
resident per shard, not both. Per card: half lanes ~8.25 GB + KV half +
activations + graph objects ≤ ~11 GB — fits 15.92 GiB with margin.
Load-time needs packed transiently per shard (convert, free); the load
path must free per-shard, not after the whole table. If a path needs the
packed blob at decode (non-lane route), that shard stays packed and skips
the lane copy — the budget is per-representation, never both.

## Phases

P0 — probes (unchanged in substance): per-card true-cold BW for the split
ratio (9070 XT vs 9060 XT — weighted, not 50/50); P2P read BW; peer-add
latency at hidden 5120 f32 via M6 ring vs RCCL (RCCL all-reduce is
graph-capturable per the earlier skill confirmation). NEW: measure the
per-rendezvous cost — it feeds the Stage-B target directly.

P1 — tensor-granular placement: (layer, tensor, half) shards in
plan_layer_devices; lanes-replace-packed per shard; conv-state channel
slice map; test consumers of layer-granular placement enumerated and
migrated. Gate: load completes, both cards ≤ 11 GB, zero managed.

P2 — TP GEMV wiring: per-card decode graphs in lockstep; M6 handoff ring
after EVERY row-parallel arm (down_proj, o_proj on attention layers, KDA
output proj) — that is 2-3 rendezvous/layer, **~130-190/token**, not one.
Act quantization runs per card on the replicated input. Short-K idle-lane
mapping fixed in the shard kernels (o_proj K=3072, down K=8704 per card —
not lane-group multiples; idle lanes must be predicated, not wrapped).

P3 — numerics gates: unchanged (peer-add reorders sums — tolerance parity
vs single-card eager, greedy 256-token window).

P4 — measurement, POST-STAGE-A: wall vs the then-current serial baseline,
per-card utilization, target = predeclared from Stage-B's formula. If the
formula says TP does not beat serial post-A, STOP and record the verdict —
the mirror elimination and idle-9060 arguments stand on their own but do
not justify wall-target engineering.

## Out of scope until P4 passes (enumerated, from review)

Non-greedy sampling (full logits merge ~1 MB/token per handoff), prefill TP
(decode graphs only; prefill stays serial-split), eager fallback semantics
per shard, MTP draft consumer, mmproj.

## Risks (REV 1 items that stand)

Cross-device lockstep skew bounded by per-layer ring sync; capture time and
graph-object VRAM roughly double; KDA state split is per-head exact; uneven
cards handled by the P0-weighted ratio (config change, not code).
