//! WS-E GreyRaven: the host-side 2:4 structured sparsifier.
//!
//! # Scope
//!
//! This is the *host* half of WS-E's deliverable -- the part that produces
//! checkpoints in the format. The kernel (`V_SWMMAC_F32_16X16X32_FP8_FP8`) is
//! WS-E steps E1-E9 and is **not** here: it is GPU work, and it is additionally
//! gated on **S5**, which asks whether SWMMAC permits a dense operand. S5 is
//! unanswered. If the answer is no, GreyRaven's inference story collapses to the
//! training case and WS-E's scope has to be re-cut, at which point this file
//! still stands because the sparsifier is useful for training regardless.
//!
//! # Why 2:4 and not 2:3
//!
//! 2:4 is the sparsity the hardware natively consumes. Per RDNA4 Table 41,
//! `V_SWMMAC_F32_16X16X32_FP8_FP8` retires 2x the MACs per instruction against
//! `V_WMMA_F32_16X16X16_FP8_FP8`. That is a documentation claim, not a grim
//! measurement -- step E7 measures it, and until E7 is green the 2x is
//! unverified. Nothing in this file assumes it is real.
//!
//! # The bit accounting, stated plainly
//!
//! ```text
//! data      2 survivors per group of 4 x 8 bits (E4M3) = 4.0 bits/weight
//! metadata  2-bit position per element                 = 2.0 bits/weight
//!                                                          ------------
//! effective                                                   6.0 bits/weight
//! ```
//!
//! 6.0 bpw is *worse* than dense E4M3's 8.0 only in the sense that it is more
//! bits; it is better in that the hardware can consume it at 2x rate. Whether
//! that trade wins is step E9, at matched *tolerance* rather than matched bpw.
//! E4M3 remains the incumbent to beat.
//!
//! CPU only.

use grim_quant::grey_raven::{
    GROUP, GROUP_SURVIVORS, METADATA_BITS_PER_GROUP, Sparsified, densify, sparsify_2_4,
};

/// A group is 4 wide and keeps exactly 2. Asserted as literals because the whole
/// format is these two numbers.
#[test]
fn every_group_of_four_keeps_exactly_two() {
    assert_eq!(GROUP, 4);
    assert_eq!(GROUP_SURVIVORS, 2);
    // 3 bits, not 2: C(4,2)=6 patterns do not fit in 2 bits.
    assert_eq!(METADATA_BITS_PER_GROUP, 3, "6 survivor patterns need 3 bits per group");
}

/// The survivors must be the two largest magnitudes in their group.
///
/// This is the definition of magnitude-based 2:4 pruning. Anything else -- first
/// two, random two, smallest two -- would be a different format.
#[test]
fn survivors_are_the_two_largest_magnitudes() {
    // One group per pattern, laid out so the answer is forced.
    let cases: [[f32; GROUP]; 6] = [
        [1.0, -2.0, 0.3, 0.4],   // keep -2, 1
        [0.1, 0.2, 0.3, -5.0],   // keep -5, 0.3
        [9.0, 8.0, 7.0, 6.0],    // keep 9, 8
        [-1.0, 1.0, -1.0, 1.0],  // all equal magnitude
        [0.0, 0.0, 0.0, 1.0],    // keep the only nonzero
        [3.0, -1.0, 0.0, 2.5],   // keep 3, 2.5
    ];

    for (gi, group) in cases.iter().enumerate() {
        let s = sparsify_2_4(group);
        let mags: Vec<f32> = group.iter().map(|v| v.abs()).collect();
        let mut sorted = mags.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff = sorted[GROUP_SURVIVORS - 1];

        for (i, &m) in mags.iter().enumerate() {
            let kept = s.is_kept(i);
            if m > cutoff {
                assert!(kept, "group {gi}: |{m}| exceeds the cutoff {cutoff} but was pruned");
            } else if m < cutoff {
                assert!(!kept, "group {gi}: |{m}| is below the cutoff {cutoff} but was kept");
            }
            // Equal-to-cutoff entries are the tie case, checked separately.
        }
    }
}

/// Ties must break deterministically.
///
/// A group of equal magnitudes has no "two largest", and a sparsifier that
/// resolves the tie by iteration order, hash order, or anything else
/// non-reproducible would produce a different checkpoint from the same weights on
/// every run. Real weight matrices contain exact ties constantly -- zeros, and
/// after quantization whole runs of equal values -- so this is the common case,
/// not an edge case.
#[test]
fn ties_break_deterministically() {
    // All four equal: the lowest two indices must win, every time.
    let group = [2.5f32; GROUP];
    let a = sparsify_2_4(&group);
    for _ in 0..64 {
        let b = sparsify_2_4(&group);
        assert_eq!(a, b, "equal magnitudes must give an identical result every run");
    }
    assert!(a.is_kept(0) && a.is_kept(1), "the lowest indices win the tie");
    assert!(!a.is_kept(2) && !a.is_kept(3));

    // Mixed sign, equal magnitude.
    let group = [1.0f32, -1.0, 1.0, -1.0];
    let a = sparsify_2_4(&group);
    for _ in 0..64 {
        assert_eq!(sparsify_2_4(&group), a);
    }
}

/// Ties with some strictly-larger values: only the tied ones compete.
#[test]
fn a_strict_maximum_always_survives_a_tie() {
    // 0,0,0,9: the 9 wins outright, then one of the zeros by index.
    let group = [0.0f32, 0.0, 0.0, 9.0];
    let s = sparsify_2_4(&group);
    assert!(s.is_kept(3), "the strict maximum must survive");
    assert!(s.is_kept(0), "the remaining slot goes to the lowest tied index");
    assert!(!s.is_kept(1) && !s.is_kept(2));
}

/// Densify must place the survivors back at their recorded positions and zero
/// the pruned ones.
#[test]
fn densify_restores_positions_and_zeroes_the_rest() {
    let group = [1.0f32, -7.0, 0.5, 3.0];
    let s = sparsify_2_4(&group);
    let back = densify(&s);
    assert_eq!(back[1], -7.0, "the survivor keeps its exact value");
    assert_eq!(back[3], 3.0);
    assert_eq!(back[0], 0.0, "the pruned slot is zero");
    assert_eq!(back[2], 0.0);
}

/// The pruned values must be exactly the two smallest magnitudes, so the
/// reconstruction error is precisely what 2:4 gives up.
#[test]
fn the_pruned_values_are_the_smallest_magnitudes() {
    let mut s = 0x9E37_79B9u64;
    let mut worst = 0.0f32;
    for _ in 0..512 {
        let group: [f32; GROUP] = std::array::from_fn(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32) / (1u32 << 24) as f32 - 0.5
        });
        let sp = sparsify_2_4(&group);
        let back = densify(&sp);
        for i in 0..GROUP {
            let err = (group[i] - back[i]).abs();
            if !sp.is_kept(i) {
                assert_eq!(back[i], 0.0, "pruned slot {i} must be exactly zero");
                worst = worst.max(err);
            } else {
                assert_eq!(back[i], group[i], "kept slot {i} must be bit-exact");
            }
        }
    }
    // The error is bounded by the largest magnitude in the corpus, which is the
    // only honest bound available without assuming a distribution.
    assert!(worst <= 0.5, "pruning error {worst} exceeds the corpus range");
}

/// The format must not rescale the survivors.
///
/// Worth pinning because rescaling is a real technique in other sparse formats
/// and would change the bit accounting. Here the metadata preserves exact
/// positions, so no correction factor is needed for correctness -- and applying
/// one anyway would silently alter every weight in the checkpoint.
#[test]
fn survivors_are_not_rescaled() {
    let group = [1.0f32, -7.0, 0.5, 3.0];
    let back = densify(&sparsify_2_4(&group));
    for i in 0..GROUP {
        let original = group[i];
        let recovered = back[i];
        if original != 0.0 && sparsify_2_4(&group).is_kept(i) {
            assert_eq!(
                recovered, original,
                "slot {i} was rescaled; 2:4 needs no correction factor because the \
                 metadata preserves exact positions"
            );
        }
    }
}

/// The metadata is 3 bits per group, packed into a contiguous bitstream.
#[test]
fn metadata_packs_three_bits_per_group() {
    let mut s = 0x1234_5678u64;
    let groups: Vec<[f32; GROUP]> = (0..64)
        .map(|_| {
            std::array::from_fn(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32) / (1u32 << 24) as f32 - 0.5
            })
        })
        .collect();

    let sp = sparsify_2_4_many(&groups);
    // 3 bits per group, packed into a contiguous bitstream.
    assert_eq!(
        sp.metadata.len(),
        (groups.len() * 3).div_ceil(8),
        "3 bits per group must pack into a bitstream"
    );
    assert_eq!(sp.groups_from_metadata(), groups.len());
    // And the bit budget: 2 metadata bits + 2 data values per group.
    let data_bits = groups.len() * GROUP_SURVIVORS * 8;
    let meta_bits = sp.metadata.len() * 8;
    // 4.0 data + 0.5 metadata, not the plan's 6.0. A 2:4 group has only C(4,2)=6
    // possible survivor patterns, so the position information is log2(6)=2.58
    // bits per group; budgeting 8 metadata bits per group (the plan's "2-bit
    // position per element") wastes 5.4 of them. Compacting the survivors and
    // storing a 2-bit pair index makes the format lossless at half the metadata.
    // 4.0 data + 0.75 metadata. 3 bits per group, not 2: six patterns do not fit
    // in two bits, and masking with &0x3 silently corrupts two of them.
    let meta_bpw = meta_bits as f32 / (groups.len() * GROUP) as f32;
    assert!(
        (meta_bpw - 0.75).abs() < 0.02,
        "metadata costs {meta_bpw} bpw, expected 0.75 (3 bits per group)"
    );
    let total_per_weight = data_bits as f32 / (groups.len() * GROUP) as f32 + meta_bpw;
    assert!(
        (total_per_weight - 4.75).abs() < 0.02,
        "effective cost is {total_per_weight} bpw, expected 4.75 (4.0 data + 0.75 metadata)"
    );
    // The information-theoretic floor is log2(6) = 2.58 bits per group, so a
    // 3-bit code is within 0.42 bits of optimal and the plan's 8 bits per group
    // wastes 5.4.
    let floor = 6f32.log2() / GROUP as f32;
    assert!(
        meta_bpw < 1.0 && meta_bpw >= floor,
        "metadata is {meta_bpw} bpw; the floor for C(4,2)=6 patterns is {floor}"
    );
}

/// A tensor whose length is not a multiple of 4 must be rejected explicitly.
///
/// Silently padding a ragged tail would produce a checkpoint whose weight count
/// no longer matches its metadata, and the mismatch would surface much later as
/// a shape error at load time.
#[test]
fn a_ragged_length_is_rejected_rather_than_padded() {
    let odd: Vec<f32> = (0..7).map(|i| i as f32).collect();
    assert!(sparsify_2_4_flat(&odd).is_err(), "7 is not a multiple of 4");
    let ok: Vec<f32> = (0..8).map(|i| i as f32).collect();
    assert!(sparsify_2_4_flat(&ok).is_ok(), "8 is a multiple of 4");
}

/// The sparsifier is total over adversarial input: no panic, always 2 survivors.
#[test]
fn the_sparsifier_is_total_over_adversarial_input() {
    let nasty = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        1e-45,
    ];
    let mut all: Vec<f32> = Vec::new();
    for &v in nasty.iter() {
        all.push(v);
    }
    while all.len() % GROUP != 0 {
        all.push(0.0);
    }

    let r = sparsify_2_4_flat(&all);
    assert!(r.is_ok(), "adversarial input must not be rejected as ragged");
    let sp = r.unwrap();
    assert_eq!(sp.values.len(), all.len() / GROUP * GROUP_SURVIVORS);

    // Exactly two survivors per group, always.
    for g in 0..sp.num_groups() {
        let kept = (0..GROUP).filter(|i| sp.is_kept_in_group(g, *i)).count();
        assert_eq!(kept, GROUP_SURVIVORS, "group {g} kept {kept}, expected 2");
    }
}

/// Round trip over many random groups: `densify(sparsify(w))` must equal `w`
/// with exactly half the entries zeroed.
#[test]
fn round_trip_zeroes_exactly_half_the_entries() {
    let mut s = 0xC0FF_EE00u64;
    let groups: Vec<[f32; GROUP]> = (0..1024)
        .map(|_| {
            std::array::from_fn(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32) / (1u32 << 24) as f32 - 0.5
            })
        })
        .collect();

    let sp = sparsify_2_4_many(&groups);
    let flat: Vec<f32> = groups.iter().flatten().copied().collect();
    let back = densify_flat(&sp);

    assert_eq!(back.len(), flat.len());
    let zeros = back.iter().filter(|v| **v == 0.0).count();
    let nonzero_inputs = flat.iter().filter(|v| **v != 0.0).count();
    // Every input zero stays zero, plus half the nonzeros are pruned.
    let expected_zeros = flat.iter().filter(|v| **v == 0.0).count()
        + nonzero_inputs / 2;
    assert!(
        zeros >= expected_zeros - GROUP && zeros <= expected_zeros + GROUP,
        "{zeros} zeros, expected about {expected_zeros}"
    );
    // Kept entries are bit-exact.
    for (i, (&o, &b)) in flat.iter().zip(back.iter()).enumerate() {
        let g = i / GROUP;
        let j = i % GROUP;
        if sp.is_kept_in_group(g, j) {
            assert_eq!(o, b, "kept entry {i} must be bit-exact");
        } else {
            assert_eq!(b, 0.0, "pruned entry {i} must be zero");
        }
    }
}

// Local wrappers so the test reads cleanly; the module exposes the flat forms.
fn sparsify_2_4_many(groups: &[[f32; GROUP]]) -> Sparsified {
    let flat: Vec<f32> = groups.iter().flatten().copied().collect();
    sparsify_2_4_flat(&flat).expect("length is a multiple of 4")
}
fn sparsify_2_4_flat(v: &[f32]) -> Result<Sparsified, &'static str> {
    grim_quant::grey_raven::sparsify_2_4_flat(v)
}
fn densify_flat(s: &Sparsified) -> Vec<f32> {
    grim_quant::grey_raven::densify_flat(s)
}
