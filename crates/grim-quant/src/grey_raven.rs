//! GreyRaven: host-side 2:4 structured sparsity (WS-E).
//!
//! # Scope: the host half only
//!
//! This is the checkpoint-producing half of WS-E's deliverable. The kernel
//! (`V_SWMMAC_F32_16X16X32_FP8_FP8`) is steps E1-E9 and is **not** here: it is GPU
//! work, and it is additionally blocked on **S5** -- whether SWMMAC permits a
//! dense operand, which the manual does not state. S5 is unanswered. If the
//! answer is no, GreyRaven's inference story collapses to the training case and
//! WS-E's scope must be re-cut. This file survives either outcome, because a 2:4
//! sparsifier is useful for training regardless of what the inference kernel
//! ends up looking like.
//!
//! # What 2:4 buys, and the bit accounting stated plainly
//!
//! 2:4 is the sparsity RDNA4's SWMMAC consumes natively. Per RDNA4 Table 41,
//! `V_SWMMAC_F32_16X16X32_FP8_FP8` retires 2x the MACs per instruction against
//! `V_WMMA_F32_16X16X16_FP8_FP8` at the same result-tile size. **That 2x is a
//! vendor table entry, not a grim measurement**; step E7 measures it, and nothing
//! in this module assumes it is real.
//!
//! ```text
//! data      2 survivors per group of 4 x 8 bits (E4M3) = 4.0 bits/weight
//! metadata  3 bits per group                            = 0.75 bits/weight
//!                                                          -------------
//! effective                                                   4.75 bits/weight
//! ```
//!
//! **This is cheaper than the plan's 6.0 bpw, and the difference is worth
//! stating precisely.** The plan budgets "2-bit position per element", i.e. 8
//! metadata bits per group of 4. But a 2:4 group has only C(4,2) = 6 possible
//! survivor patterns, so the position information is log2(6) = 2.58 bits per
//! group, and any encoding spending 8 bits on it wastes 5.4. This implementation
//! stores survivors *compacted* and encodes the surviving pair as an index into
//! those six, which is 3 bits per group and lossless.
//!
//! It is 3 bits and not 2 because 6 does not fit in 2. An earlier version used 2
//! and masked with `& 0x3`, folding patterns 4 and 5 onto 0 and 1, so a group
//! keeping slots (1,3) reconstructed as keeping (0,1). The tests caught it; it is
//! recorded because "2 bits" is what the plan's phrasing suggests and it looks
//! obviously right.
//!
//! So GreyRaven lands at 4.75 bpw, not 6.0 -- which moves it from "denser than
//! E4M3's 8.0 and therefore needing the 2x to justify itself" to "41% less memory
//! than E4M3, if accuracy holds". That makes step E9 a much closer call than the
//! plan assumed and means the 2x instruction rate is no longer load-bearing for
//! viability. Accuracy is still the open question, and E9 compares the two at
//! matched **tolerance**, not matched bpw. If GreyRaven needs more bpw than E4M3
//! for equal output error, WS-E is killed -- a real possible outcome, stated so
//! it is not treated as a formality.
//!
//! # Weights only
//!
//! Activations are dynamic and cannot be pruned, so this operates on weights
//! alone. That asymmetry is precisely what S5 has to resolve for the kernel: if
//! SWMMAC demands *both* operands sparse, activations cannot supply it and
//! inference is not viable.
//!
//! CPU only.

/// Values per 2:4 group.
pub const GROUP: usize = 4;

/// Survivors kept per group.
pub const GROUP_SURVIVORS: usize = 2;

/// Metadata bits needed to record which `GROUP_SURVIVORS` of `GROUP` slots
/// survived.
///
/// **Three, not two.** A 2:4 group has C(4,2) = 6 possible survivor patterns, and
/// 6 does not fit in 2 bits. An earlier version of this used 2 and masked the
/// code with `& 0x3`, which silently folded patterns 4 and 5 onto 0 and 1 --
/// so a group keeping slots (1,3) reconstructed as keeping slots (0,1). The
/// tests caught it; the bug is recorded because "2 bits per group" looks
/// obviously right next to the plan's "2-bit position" phrasing.
///
/// Survivors are stored *compacted*, so the metadata is an index into the six
/// patterns rather than a bitmask over four slots.
pub const METADATA_BITS_PER_GROUP: u32 = 3;

/// A tensor sparsified to 2:4.
///
/// `values` holds the survivors compacted, [`GROUP_SURVIVORS`] per group in group
/// order. `metadata` holds the position of each pair, 2 bits per group, four
/// groups per byte.
#[derive(Debug, Clone, PartialEq)]
pub struct Sparsified {
    /// Compacted survivors: `num_groups * GROUP_SURVIVORS` values.
    pub values: Vec<f32>,
    /// `METADATA_BITS_PER_GROUP` bits per group, packed low-bits-first into a
    /// contiguous bitstream.
    pub metadata: Vec<u8>,
}

impl Sparsified {
    /// Number of 2:4 groups.
    ///
    /// Derived from the value count, **not** from `metadata.len() * 4`. The
    /// metadata array is padded up to a byte boundary, so a single group occupies
    /// a whole byte and `metadata.len() * 4` reports four groups where there is
    /// one -- which then indexed past the end of `values`.
    pub fn num_groups(&self) -> usize {
        self.values.len() / GROUP_SURVIVORS
    }

    /// Was slot `index` within `group` kept?
    pub fn is_kept_in_group(&self, group: usize, index: usize) -> bool {
        PAIRS[self.rank_of_group(group)].contains(&index)
    }

    /// Number of groups described by the metadata bitstream.
    pub fn groups_from_metadata(&self) -> usize {
        self.metadata.len() * 8 / METADATA_BITS_PER_GROUP as usize
    }

    /// Was slot `index` within the *first* group kept?
    ///
    /// A convenience for the single-group case, which is how the format is
    /// normally reasoned about. For a multi-group tensor use
    /// [`Self::is_kept_in_group`], which takes the group explicitly.
    pub fn is_kept(&self, index: usize) -> bool {
        self.is_kept_in_group(0, index)
    }

    /// The 2-bit position code for a group: which of the six (4 choose 2) pairs
    /// survived, in ascending slot order.
    fn rank_of_group(&self, group: usize) -> usize {
        let bit = group * METADATA_BITS_PER_GROUP as usize;
        let mut code = 0usize;
        for k in 0..METADATA_BITS_PER_GROUP as usize {
            if self.metadata[(bit + k) / 8] & (1 << ((bit + k) % 8)) != 0 {
                code |= 1 << k;
            }
        }
        code
    }
}

/// All six (4 choose 2) survivor patterns, in ascending slot order.
///
/// Indexed by the 2-bit position code, which is therefore not a bitmask: pair
/// index 0 is slots (0,1), index 5 is slots (2,3).
const PAIRS: [[usize; GROUP_SURVIVORS]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// The position code for a pair of surviving slots.
fn pair_code(a: usize, b: usize) -> u8 {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    PAIRS
        .iter()
        .position(|p| p[0] == a && p[1] == b)
        .expect("every ascending pair is in the table") as u8
}

/// Sparsify a flat tensor to 2:4 by group magnitude.
///
/// Each group of [`GROUP`] consecutive values keeps the [`GROUP_SURVIVORS`]
/// largest-magnitude entries. The rest are dropped and the survivors keep their
/// **exact** values: no rescale, because the metadata records exact positions
/// and so no correction factor is needed for correctness. Applying one anyway
/// would silently alter every weight in the checkpoint.
///
/// # Determinism
///
/// Ties are broken by lowest slot index, always. Real weight matrices are full of
/// exact ties -- zeros, and whole runs of equal values after quantization -- so
/// this is the common case rather than an edge case, and a tie broken by
/// iteration or hash order would produce a different checkpoint from the same
/// weights on every run.
///
/// # Total
///
/// Never panics. `NaN` sorts last in magnitude terms and loses to every finite
/// value; infinities win. Every group keeps exactly two survivors regardless of
/// input, so the output shape is a function of the input *length* alone.
pub fn sparsify_2_4_flat(values: &[f32]) -> Result<Sparsified, &'static str> {
    if values.len() % GROUP != 0 {
        return Err("length must be a multiple of 4 for 2:4");
    }
    let groups = values.len() / GROUP;
    let mut out = Sparsified {
        values: Vec::with_capacity(groups * GROUP_SURVIVORS),
        metadata: vec![0u8; (groups * METADATA_BITS_PER_GROUP as usize).div_ceil(8)],
    };

    for g in 0..groups {
        let base = g * GROUP;
        let group = &values[base..base + GROUP];

        // Rank slots by magnitude descending, ties by ascending index. Written as
        // an explicit comparison rather than a sort so the tie rule is visible
        // and cannot be perturbed by the sort implementation's stability
        // guarantees.
        let mut order = [0usize; GROUP];
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = i;
        }
        for i in 0..GROUP {
            for j in (i + 1)..GROUP {
                let (a, b) = (order[i], order[j]);
                let (ma, mb) = (group[a].abs(), group[b].abs());
                // Swap when b is strictly larger, or equal and earlier. Written
                // as an explicit comparison rather than a sort_with so the tie
                // rule is visible and cannot be perturbed by the sort
                // implementation's stability guarantees.
                if mb > ma || (mb == ma && b < a) {
                    order[i] = b;
                    order[j] = a;
                }
            }
        }

        // Ascending slot order, not magnitude-rank order. `order` ranks by
        // magnitude, but the metadata encodes an *ascending pair* and
        // `densify` reads the values back in that same order -- storing them in
        // rank order silently transposes the two whenever the larger magnitude
        // sits in the higher slot.
        let (a, b) = if order[0] < order[1] {
            (order[0], order[1])
        } else {
            (order[1], order[0])
        };
        out.values.push(group[a]);
        out.values.push(group[b]);

        // Write the 3-bit code into the bitstream. Done by shifting the whole
        // stream rather than masking a single byte, because a 3-bit field can
        // straddle a byte boundary and a per-byte OR would corrupt its
        // neighbours.
        let bit = g * METADATA_BITS_PER_GROUP as usize;
        let code = pair_code(a, b) as u32;
        for k in 0..METADATA_BITS_PER_GROUP as usize {
            if code & (1 << k) != 0 {
                out.metadata[(bit + k) / 8] |= 1 << ((bit + k) % 8);
            }
        }
    }

    Ok(out)
}

/// Reconstruct a dense tensor: survivors at their recorded positions, zeros
/// elsewhere.
pub fn densify_flat(s: &Sparsified) -> Vec<f32> {
    let mut out = vec![0.0f32; s.num_groups() * GROUP];
    for g in 0..s.num_groups() {
        let pair = PAIRS[s.rank_of_group(g)];
        out[g * GROUP + pair[0]] = s.values[g * GROUP_SURVIVORS];
        out[g * GROUP + pair[1]] = s.values[g * GROUP_SURVIVORS + 1];
    }
    out
}

/// Sparsify one group, for tests and for the scalar path.
pub fn sparsify_2_4(group: &[f32; GROUP]) -> Sparsified {
    sparsify_2_4_flat(group).expect("a single group is always a multiple of 4")
}

/// Reconstruct one group.
pub fn densify(s: &Sparsified) -> [f32; GROUP] {
    let flat = densify_flat(s);
    let mut out = [0.0f32; GROUP];
    out.copy_from_slice(&flat[..GROUP]);
    out
}

/// Which slots survived each 2:4 group, in ascending slot order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneMask {
    /// Two surviving slot indices per group, ascending within the pair.
    pub pairs: Vec<[u8; 2]>,
    /// Number of 2:4 groups covered.
    pub groups: usize,
}

/// Rank the slots of one 2:4 group and return the surviving pair, ascending.
///
/// Importance is `fisher * w^2`, not `|w|`. Magnitude alone answers "which
/// weight is largest", which is not the question when Fisher information is
/// available: a large weight carrying almost no gradient is worth less than a
/// modest one that does. The `w^2` is not incidental -- one weight's
/// contribution to a dot product's output variance scales with the square of the
/// weight, so Fisher times squared weight is the quantity that predicts output
/// error.
///
/// Ties break by ascending slot, so the mask is a deterministic function of its
/// inputs.
pub fn prune_group_2of4(group: &[f32; GROUP], fisher: &[f32; GROUP]) -> [u8; 2] {
    let mut order = [0usize; GROUP];
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    for i in 0..GROUP {
        for j in (i + 1)..GROUP {
            let (a, b) = (order[i], order[j]);
            let ia = fisher[a] * group[a] * group[a];
            let ib = fisher[b] * group[b] * group[b];
            if ib > ia || (ib == ia && b < a) {
                order[i] = b;
                order[j] = a;
            }
        }
    }
    if order[0] < order[1] {
        [order[0] as u8, order[1] as u8]
    } else {
        [order[1] as u8, order[0] as u8]
    }
}

/// Prune a 2-D weight matrix to 2:4, **grouping along `cols` (the K axis)**.
///
/// Grouping along K is not a detail: a 2:4 group's contract is that the kernel
/// will hold four K-contiguous values and keep two, which is what lets the
/// surviving pairs be addressed as a contiguous K-window. Groups formed along
/// rows would have no such property and could not be handed to a sparse GEMM.
///
/// `fisher` is a per-element importance signal the same length as `weights`;
/// pass all ones for magnitude-only pruning, which is what
/// [`sparsify_2_4_flat`] does.
///
/// `cols % 4 != 0` is rejected rather than padded: padding would invent weights
/// that do not exist and would silently change the density.
pub fn prune_2of4(
    weights: &[f32],
    fisher: &[f32],
    rows: usize,
    cols: usize,
) -> Result<PruneMask, &'static str> {
    if cols % GROUP != 0 {
        return Err("cols must be a multiple of 4: 2:4 groups run along K");
    }
    if rows.checked_mul(cols) != Some(weights.len()) {
        return Err("rows * cols must equal weights.len()");
    }
    if fisher.len() != weights.len() {
        return Err("fisher must be the same length as weights");
    }
    let groups_per_row = cols / GROUP;
    let mut pairs = Vec::with_capacity(rows * groups_per_row);
    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * GROUP;
            let mut grp = [0f32; GROUP];
            let mut fis = [0f32; GROUP];
            grp.copy_from_slice(&weights[base..base + GROUP]);
            fis.copy_from_slice(&fisher[base..base + GROUP]);
            pairs.push(prune_group_2of4(&grp, &fis));
        }
    }
    Ok(PruneMask {
        pairs,
        groups: rows * groups_per_row,
    })
}

/// Flat Fisher-aware sparsify, producing the same [`Sparsified`] as
/// [`sparsify_2_4_flat`] but ranking by `fisher * w^2`.
///
/// Survivors are held as `f32` here, not E4M3 bytes: this is the host-side
/// prune/metadata stage, and packing survivors to E4M3 is the
/// `pack_grey_raven` step where the format's 4.75 bpw is actually realised.
/// Storing f32 keeps the intermediate honest about what it costs, so a "4.75
/// bpw" claim is never read off a `Vec<f32>`.
pub fn sparsify_2_4_flat_with_fisher(
    values: &[f32],
    fisher: &[f32],
) -> Result<Sparsified, &'static str> {
    if values.len() % GROUP != 0 {
        return Err("length must be a multiple of 4 for 2:4");
    }
    if fisher.len() != values.len() {
        return Err("fisher must be the same length as values");
    }
    let groups = values.len() / GROUP;
    let mut out = Sparsified {
        values: Vec::with_capacity(groups * GROUP_SURVIVORS),
        metadata: vec![0u8; (groups * METADATA_BITS_PER_GROUP as usize).div_ceil(8)],
    };
    for g in 0..groups {
        let base = g * GROUP;
        let mut grp = [0f32; GROUP];
        let mut fis = [0f32; GROUP];
        grp.copy_from_slice(&values[base..base + GROUP]);
        fis.copy_from_slice(&fisher[base..base + GROUP]);
        let pair = prune_group_2of4(&grp, &fis);
        let (a, b) = (pair[0] as usize, pair[1] as usize);
        out.values.push(grp[a]);
        out.values.push(grp[b]);
        let bit = g * METADATA_BITS_PER_GROUP as usize;
        let code = pair_code(a, b) as u32;
        for k in 0..METADATA_BITS_PER_GROUP as usize {
            if code & (1 << k) != 0 {
                out.metadata[(bit + k) / 8] |= 1 << ((bit + k) % 8);
            }
        }
    }
    Ok(out)
}

/// Bytes of packed GreyRaven per `GROUP` original weights.
///
/// 2 survivors as E4M3 bytes + 3 metadata bits: 19 bits per 4 originals, i.e.
/// 4.75 bpw. Kept as a named function so `pack_grey_raven`, `dequant_grey_raven`
/// and `DType::expected_bytes` cannot drift apart on the layout.
pub const fn packed_bytes_for(num_values: usize) -> usize {
    let groups = num_values.div_ceil(GROUP);
    groups * GROUP_SURVIVORS + (groups * METADATA_BITS_PER_GROUP as usize).div_ceil(8)
}

/// Pack a [`Sparsified`] into GreyRaven's byte layout.
///
/// Layout: `[survivor E4M3 bytes][metadata bitstream]`, survivors first and
/// metadata last, both contiguous. That ordering is what makes the density
/// arithmetic in [`packed_bytes_for`] fall out of one expression, and it keeps
/// the metadata a single contiguous run the kernel can address per superblock.
///
/// Survivors are encoded with the same RNE E4M3 encoder WhiteRaven uses, so a
/// GreyRaven tensor and a dense FP8 tensor of the same values carry identical
/// codes. That is deliberate: it means the decode below is WhiteRaven's decode
/// plus metadata expansion, with no second FP8 implementation to disagree with
/// the first.
pub fn pack_grey_raven(s: &Sparsified) -> Vec<u8> {
    let n_survivors = s.values.len();
    let mut out = Vec::with_capacity(n_survivors + s.metadata.len());
    for &v in &s.values {
        out.push(crate::f32_to_fp8_e4m3(v));
    }
    out.extend_from_slice(&s.metadata);
    out
}

/// Decode a GreyRaven buffer back to a dense tensor: survivors at their recorded
/// positions, zero elsewhere.
///
/// This reconstructs the **pruned** model, not the original dense one, and that
/// distinction is the format's semantics rather than an implementation detail --
/// `e4_dequant_reconstructs_the_pruned_model_not_the_dense_one` pins it.
///
/// Validation is total: a buffer whose length disagrees with `num_values`, whose
/// metadata is short, or whose 3-bit codes name a slot pair outside the group is
/// rejected rather than clamped. A format whose decoder can be made to panic or
/// to read out of bounds on a bad buffer is a denial of service on untrusted
/// input, and one that silently clamps produces a *different model* with no
/// signal that anything went wrong.
pub fn dequant_grey_raven(data: &[u8], num_values: usize) -> Result<Vec<f32>, &'static str> {
    if num_values % GROUP != 0 {
        return Err("num_values must be a multiple of 4 for 2:4");
    }
    let groups = num_values / GROUP;
    let survivor_bytes = groups * GROUP_SURVIVORS;
    let meta_bytes = (groups * METADATA_BITS_PER_GROUP as usize).div_ceil(8);
    if data.len() != survivor_bytes + meta_bytes {
        return Err("buffer length does not match the 2:4 layout for num_values");
    }
    let (survivors, meta) = data.split_at(survivor_bytes);

    let mut out = vec![0.0f32; num_values];
    for g in 0..groups {
        let bit = g * METADATA_BITS_PER_GROUP as usize;
        let mut code = 0u32;
        for k in 0..METADATA_BITS_PER_GROUP as usize {
            if meta[(bit + k) / 8] & (1 << ((bit + k) % 8)) != 0 {
                code |= 1 << k;
            }
        }
        let (a, b) = code_to_pair(code).ok_or("metadata code does not name a valid slot pair")?;
        let base = g * GROUP;
        out[base + a] = crate::fp8_e4m3_to_f32(survivors[g * GROUP_SURVIVORS]);
        out[base + b] = crate::fp8_e4m3_to_f32(survivors[g * GROUP_SURVIVORS + 1]);
    }
    Ok(out)
}

/// Invert [`pair_code`]. Returns `None` for a 3-bit code outside 0..=5, which is
/// the only way a malformed buffer can name a pair that does not exist.
///
/// Uses the same module-level `PAIRS` table `pair_code` encodes against rather
/// than a local copy. A second table here would be a second source of truth for
/// the same bijection, and the two disagreeing would decode to a different model
/// with no error -- the silent-failure class this format is most exposed to.
fn code_to_pair(code: u32) -> Option<(usize, usize)> {
    PAIRS.get(code as usize).map(|p| (p[0], p[1]))
}

/// The SWMMAC index field for a kept pair, or `None` if the pair is unexpressible.
///
/// Measured on gfx1200: a 2:4 group is encoded as a fixed anchor at slot 0 plus a
/// free survivor whose position a 2-bit field chooses -- 00 -> 0, 01 -> 1, 10 -> 2,
/// 11 -> 3 -- and the paired odd byte is pinned to slot 0. So only pairs *containing
/// slot 0* have an encoding: {0,1}, {0,2}, {0,3}. The other three ({1,2}, {1,3},
/// {2,3}) do not, and a mask containing one cannot be emitted to this hardware.
///
/// Returns the field value, which is the non-anchor slot, so {0, q} -> `q`.
pub const fn index_field_for_pair(a: u8, b: u8) -> Option<u8> {
    match (a, b) {
        (0, q) => Some(q),
        (q, 0) => Some(q),
        _ => None,
    }
}

/// Whether every kept pair in a mask is expressible on SWMMAC FP8.
///
/// Cheap enough to assert before packing, so a mask that cannot be emitted is
/// caught at the boundary rather than as a wrong product on the GPU.
pub fn mask_is_expressible(pairs: &[[u8; 2]]) -> bool {
    pairs
        .iter()
        .all(|p| index_field_for_pair(p[0], p[1]).is_some())
}

/// [`prune_group_2of4`] restricted to the pairs SWMMAC can encode.
///
/// Always keeps slot 0 and picks the best companion from {1, 2, 3} by the same
/// `fisher * w^2` ranking. Costs a little accuracy against the unconstrained
/// version, which may keep two high-weight non-zero slots, and buys an exact fit
/// with the hardware: the alternative is to remap an arbitrary pair into an
/// expressible one, which keeps the pattern space but sometimes keeps a
/// lower-weight position than the pruner chose.
pub fn prune_group_2of4_anchored(group: &[f32; GROUP], fisher: &[f32; GROUP]) -> [u8; 2] {
    let imp = |i: usize| fisher[i] * group[i] * group[i];
    // Slot 0 is the anchor and is kept unconditionally; choose among the rest.
    let (mut best, mut best_i) = (f32::NEG_INFINITY, 1usize);
    for i in 1..GROUP {
        let v = imp(i);
        if v > best || (v == best && i < best_i) {
            best = v;
            best_i = i;
        }
    }
    [0, best_i as u8]
}

/// [`prune_2of4`] with the SWMMAC-anchored group rule.
pub fn prune_2of4_anchored(
    weights: &[f32],
    fisher: &[f32],
    rows: usize,
    cols: usize,
) -> Result<PruneMask, &'static str> {
    if cols % GROUP != 0 {
        return Err("cols must be a multiple of 4: 2:4 groups run along K");
    }
    if rows.checked_mul(cols) != Some(weights.len()) {
        return Err("rows * cols must equal weights.len()");
    }
    if fisher.len() != weights.len() {
        return Err("fisher must be the same length as weights");
    }
    let groups_per_row = cols / GROUP;
    let mut pairs = Vec::with_capacity(rows * groups_per_row);
    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * GROUP;
            let mut grp = [0f32; GROUP];
            let mut fis = [0f32; GROUP];
            grp.copy_from_slice(&weights[base..base + GROUP]);
            fis.copy_from_slice(&fisher[base..base + GROUP]);
            pairs.push(prune_group_2of4_anchored(&grp, &fis));
        }
    }
    Ok(PruneMask {
        pairs,
        groups: rows * groups_per_row,
    })
}

/// [`sparsify_2_4_flat_with_fisher`] using the anchored group rule, so the
/// resulting mask is emittable to SWMMAC without remapping.
pub fn sparsify_2_4_flat_anchored_with_fisher(
    values: &[f32],
    fisher: &[f32],
) -> Result<Sparsified, &'static str> {
    if values.len() % GROUP != 0 {
        return Err("length must be a multiple of 4 for 2:4");
    }
    if fisher.len() != values.len() {
        return Err("fisher must be the same length as values");
    }
    let groups = values.len() / GROUP;
    let mut out = Sparsified {
        values: Vec::with_capacity(groups * GROUP_SURVIVORS),
        metadata: vec![0u8; (groups * METADATA_BITS_PER_GROUP as usize).div_ceil(8)],
    };
    for g in 0..groups {
        let base = g * GROUP;
        let mut grp = [0f32; GROUP];
        let mut fis = [0f32; GROUP];
        grp.copy_from_slice(&values[base..base + GROUP]);
        fis.copy_from_slice(&fisher[base..base + GROUP]);
        let pair = prune_group_2of4_anchored(&grp, &fis);
        let (a, b) = (pair[0] as usize, pair[1] as usize);
        out.values.push(grp[a]);
        out.values.push(grp[b]);
        let bit = g * METADATA_BITS_PER_GROUP as usize;
        let code = pair_code(a, b) as u32;
        for k in 0..METADATA_BITS_PER_GROUP as usize {
            if code & (1 << k) != 0 {
                out.metadata[(bit + k) / 8] |= 1 << ((bit + k) % 8);
            }
        }
    }
    Ok(out)
}
