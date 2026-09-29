//! GreyRaven (WS-E) steps E0–E3: the format's identity, its prune, and its
//! cost model.
//!
//! These are the four steps that need no GPU. E4 onward need a
//! `V_SWMMAC_F32_16X16X32_FP8_FP8` kernel, whose exact builtin signature S5
//! recorded as undetermined, so nothing above E3 could be written honestly yet
//! -- a kernel authored against an unknown intrinsic is how you get a green
//! compile and a wrong answer.
//!
//! E3 is named `effective_bpw_is_4_75`, not the plan's `effective_bpw_is_six`.
//! The plan's 6.0 bpw assumes 2 metadata bits per 2:4 group, but a group has
//! C(4,2) = 6 possible survivor patterns and 6 does not fit in 2 bits -- the
//! same defect the module doc on `METADATA_BITS_PER_GROUP` records, where a
//! 2-bit code silently folded patterns 4 and 5 onto 0 and 1. The real figure is
//! 2 survivors x 8 bits + 3 metadata bits per 4 originals = 19/4 = 4.75 bpw.
//! Asserting 6.0 would pin a cost model known to be wrong, which is the
//! opposite of what E3 exists to do.

use grim_quant::grey_raven::{
    densify_flat, dequant_grey_raven, pack_grey_raven, packed_bytes_for, prune_2of4,
    prune_group_2of4, sparsify_2_4_flat_with_fisher, GROUP, GROUP_SURVIVORS,
    METADATA_BITS_PER_GROUP,
};
use grim_tensor::{ArithType, BlockDtype, DType, FloatPackScheme, QuantFormat, Storage};

/// E0 — GreyRaven and WhiteRaven are different formats, not a tuning knob.
///
/// The plan is explicit that a "GreyRaven kernel" containing both formats would
/// be as wrong as a "mixed FP8/FP16 kernel" not being the WhiteRaven kernel. The
/// two differ in three independent ways -- dtype tag, operand geometry, and
/// weight layout -- and all three are pinned here so collapsing any two of them
/// into one scheme fails.
#[test]
fn e0_grey_raven_is_distinct_from_white_raven() {
    let grey = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp8Sparse24),
    };
    // WhiteRaven's own tag: dense E4M3. This is `FloatPack(Fp8)`, not
    // `Block(Fp8)` -- the ROCm dispatch accepts both spellings for the FP8 path,
    // but `QuantFormat::Fp8` round-trips to `FloatPack`, so using the Block
    // spelling here would have made the round-trip assertion below fail for a
    // reason unrelated to what E0 is about.
    let white = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(FloatPackScheme::Fp8),
    };

    // 1. Distinct tags, distinct quant formats, and the round trip preserves
    //    the distinction in both directions.
    assert_ne!(grey.storage, white.storage, "GreyRaven and WhiteRaven share a storage tag");
    assert_ne!(
        QuantFormat::try_from(&grey.storage).unwrap(),
        QuantFormat::try_from(&white.storage).unwrap(),
        "GreyRaven and WhiteRaven resolve to the same QuantFormat"
    );
    assert_eq!(
        Storage::from(QuantFormat::Fp8Sparse24),
        grey.storage,
        "Fp8Sparse24 must not round-trip through Fp8"
    );
    assert_eq!(Storage::from(QuantFormat::Fp8), white.storage);

    // 2. Distinct cost: at 4096 elements WhiteRaven is 4096 bytes of dense
    //    codes, GreyRaven is 4.75 bpw, i.e. 2432 bytes. If these ever match,
    //    one of the two has stopped describing itself.
    let elems = 4096;
    let white_bytes = white.expected_bytes(elems);
    let grey_bytes = grey.expected_bytes(elems);
    assert_eq!(white_bytes, elems, "dense E4M3 is one byte per weight");
    assert_eq!(grey_bytes, 2432, "GreyRaven at 4096 elements must be 2432 bytes");
    assert!(
        grey_bytes < white_bytes,
        "a 2:4 format cannot cost more than the dense format it prunes"
    );

    // 3. Distinct geometry, stated as the numbers the sparse kernel will use.
    //    WhiteRaven's A is 16x16; GreyRaven's B is 16x32 (half pruned slots).
    const WHITE_RAVEN_TILE: (usize, usize) = (16, 16);
    const GREY_RAVEN_SPARSE_TILE: (usize, usize) = (16, 32);
    assert_ne!(
        WHITE_RAVEN_TILE, GREY_RAVEN_SPARSE_TILE,
        "if the tiles match, one kernel could claim both and the formats are not distinct"
    );
}

/// E1 — exactly two of every four survive, and with uniform Fisher those are the
/// two largest magnitudes.
#[test]
fn e1_prune_keeps_exactly_two_of_every_four() {
    // Hand-built groups so the expected survivor set is not whatever the
    // implementation happens to produce.
    let groups: [[f32; GROUP]; 4] = [
        [1.0, 2.0, 3.0, 4.0],   // keeps 2,3
        [9.0, 1.0, 2.0, 3.0],   // keeps 0,1
        [-7.0, 5.0, -1.0, 0.5], // magnitude: 7 and 5 -> slots 0,1
        [0.25, 0.5, 0.125, 1.0],// keeps 2,3
    ];
    let uniform = [1.0f32; GROUP];

    for (gi, g) in groups.iter().enumerate() {
        let pair = prune_group_2of4(g, &uniform);
        assert_eq!(pair[0] < pair[1], true, "pair must be ascending: {pair:?}");
        let mut expected: Vec<usize> = (0..GROUP)
            .filter(|&i| g[i].abs() >= f32::MIN_POSITIVE)
            .collect();
        // Keep the two largest magnitudes, ties by ascending slot.
        expected.sort_by(|&a, &b| {
            g[b].abs().partial_cmp(&g[a].abs()).unwrap().then(a.cmp(&b))
        });
        expected.truncate(GROUP_SURVIVORS);
        expected.sort_unstable();
        assert_eq!(
            vec![pair[0] as usize, pair[1] as usize],
            expected,
            "group {gi} {g:?}: wrong survivors"
        );
    }

    // The 2-D entry point must agree with the per-group one, and group along K.
    let cols = 8;
    let rows = 3;
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37) % 13) as f32 * 0.5 - 3.0).collect();
    let f = vec![1.0f32; rows * cols];
    let mask = prune_2of4(&w, &f, rows, cols).expect("cols is a multiple of 4");
    assert_eq!(mask.groups, rows * (cols / GROUP));
    assert_eq!(mask.pairs.len(), mask.groups);
    for (gi, pair) in mask.pairs.iter().enumerate() {
        let r = gi / (cols / GROUP);
        let g = gi % (cols / GROUP);
        let base = r * cols + g * GROUP;
        let mut grp = [0f32; GROUP];
        grp.copy_from_slice(&w[base..base + GROUP]);
        assert_eq!(
            *pair,
            prune_group_2of4(&grp, &[1.0; GROUP]),
            "2-D group {gi} disagrees with the per-group pruner"
        );
    }
}

/// E2 — Fisher outranks magnitude.
///
/// This is the step that proves the importance signal is wired rather than
/// decorative. The group is built so the largest-magnitude element has the
/// *lowest* Fisher score: a magnitude-only pruner keeps it, a Fisher-aware one
/// must drop it. If this test ever passes against a magnitude-only pruner, the
/// pruner is not the one under test.
#[test]
fn e2_prune_respects_fisher_over_magnitude() {
    // Slots:          0        1       2        3
    let weights = [8.0f32, 3.0, 2.5, 0.1];
    // Slot 0 is the largest by magnitude and the least important by Fisher.
    let fisher = [0.01f32, 1.0, 1.0, 1.0];

    // Magnitude alone would keep 0 (|8|) and 1 (|3|).
    let mag_only = prune_group_2of4(&weights, &[1.0; GROUP]);
    assert_eq!(mag_only, [0, 1], "magnitude-only pruner should keep slots 0 and 1");

    // Fisher * w^2 : slot 0 -> 0.01*64 = 0.64, slots 1,2 -> 9.0, 6.25.
    let guided = prune_group_2of4(&weights, &fisher);
    assert_eq!(guided, [1, 2], "Fisher must evict the large-but-insignificant slot 0");
    assert!(
        !guided.contains(&0),
        "slot 0 has the lowest fisher*w^2 and must not survive"
    );

    // The same signal must reach the flat sparsifier, not just the group helper,
    // or the wired path and the used path diverge.
    let s = sparsify_2_4_flat_with_fisher(&weights, &fisher).expect("length is a multiple of 4");
    let dense = densify_flat(&s);
    assert_eq!(dense[0], 0.0, "pruned slot must reconstruct as zero");
    assert_eq!(dense[1], 3.0);
    assert_eq!(dense[2], 2.5);
    assert_eq!(dense[3], 0.0);
    assert_eq!(s.values.len(), GROUP_SURVIVORS);
}

/// E3 — the cost model, pinned at the number that is actually true.
#[test]
fn e3_effective_bpw_is_4_75() {
    // Per group of 4 originals: 2 survivors x 8 bits of E4M3 + 3 metadata bits
    // = 19 bits, so 19/4 = 4.75 bits per original weight.
    assert_eq!(METADATA_BITS_PER_GROUP, 3, "C(4,2)=6 patterns cannot fit in 2 bits");

    const ELEMS: usize = 4096; // 1024 groups of 4
    let groups = ELEMS / GROUP;
    let expected_bytes = groups * GROUP_SURVIVORS + (groups * METADATA_BITS_PER_GROUP as usize).div_ceil(8);
    assert_eq!(expected_bytes, 1024 * 2 + 384, "1024 groups -> 2048 survivors + 384 metadata");

    let grey = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp8Sparse24),
    };
    assert_eq!(
        grey.expected_bytes(ELEMS),
        expected_bytes,
        "DType::expected_bytes must agree with the hand-computed GreyRaven layout"
    );

    // The dtype's size model and the packer's real output must agree, at several
    // shapes including one that is not a whole number of metadata bytes' worth.
    // E3 pins the arithmetic and E4 pins the buffer; neither would notice if the
    // other drifted, and a size model that disagrees with the packer is how a
    // VRAM estimate silently under-counts a whole checkpoint.
    for &n2 in &[4usize, 8, 12, 64, 1024, 4096, 12_288] {
        assert_eq!(
            grey.expected_bytes(n2),
            packed_bytes_for(n2),
            "DType::expected_bytes and pack_grey_raven disagree at {n2} elements"
        );
    }

    // The density itself, as bits per original weight, so the number is named
    // rather than left implicit in a byte count that a future metadata change
    // could alter.
    let bpw = expected_bytes as f64 * 8.0 / ELEMS as f64;
    assert!(
        (bpw - 4.75).abs() < 1e-9,
        "GreyRaven density must be 4.75 bpw, got {bpw}"
    );

    // And the round trip: what a real packed buffer would cost, from the actual
    // sparsifier output rather than from arithmetic about it.
    let values: Vec<f32> = (0..ELEMS).map(|i| ((i * 2654435761usize) % 1000) as f32 * 0.01).collect();
    let s = sparsify_2_4_flat_with_fisher(&values, &vec![1.0; ELEMS]).expect("multiple of 4");
    let survivor_bytes = s.values.len(); // one E4M3 byte each
    let real_bytes = survivor_bytes + s.metadata.len();
    assert_eq!(real_bytes, expected_bytes, "sparsifier output must match the cost model");
}

/// E4 — dequant reconstructs the *pruned* model, not the dense one.
///
/// This is the step that turns 4.75 bpw from an accounting figure into a real
/// buffer: survivors become actual E4M3 bytes and the metadata a real bitstream.
///
/// The assertion is deliberately two-sided. Against the pruned reference the
/// decode must be tight; against the *original* dense weights it must NOT be,
/// because half of them are gone by construction. The second half is what stops
/// a later "improvement" from quietly densifying the buffer to make the
/// comparison against the dense model pass -- which would restore accuracy and
/// simultaneously destroy the format.
#[test]
fn e4_dequant_reconstructs_the_pruned_model_not_the_dense_one() {
    let n = 512usize;
    let dense: Vec<f32> = (0..n)
        .map(|i| (((i * 2654435761usize) % 977) as f32) * 0.011 - 5.0)
        .collect();
    let fisher: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { 0.01 } else { 1.0 }).collect();

    let s = sparsify_2_4_flat_with_fisher(&dense, &fisher).expect("n is a multiple of 4");
    let packed = pack_grey_raven(&s);

    // The buffer is exactly the predicted size: survivors as E4M3 bytes, then
    // the metadata bitstream.
    assert_eq!(
        packed.len(),
        packed_bytes_for(n),
        "packed buffer must match the layout cost model exactly"
    );
    assert_eq!(packed.len() * 8, n * 19 / 4, "19 bits per 4 originals = 4.75 bpw");

    let decoded = dequant_grey_raven(&packed, n).expect("well-formed buffer");

    // Pruned reference: survivors at their positions, zeros elsewhere. This is
    // what densify_flat already computes, computed independently here from the
    // E4M3 round trip so the test is not just asserting the decoder equals the
    // encoder it was written beside.
    let pruned = densify_flat(&s);
    let e4m3_pruned: Vec<f32> = pruned
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { round_e4m3(v) })
        .collect();

    let mut worst_vs_pruned = 0.0f32;
    for i in 0..n {
        worst_vs_pruned = worst_vs_pruned.max((decoded[i] - e4m3_pruned[i]).abs());
    }
    assert!(
        worst_vs_pruned < 1e-6,
        "decode must match the pruned reference, worst abs {worst_vs_pruned:.3e}"
    );

    // Half the weights are structurally zero, so the decode cannot be close to
    // the dense original.
    let zeros = decoded.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros >= n / 2,
        "a 2:4 decode must have at least half its weights structurally zero, found {zeros}/{n}"
    );
    let dense_err: f32 = decoded
        .iter()
        .zip(&dense)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dense_scale = dense.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        dense_err > 0.5 * dense_scale,
        "decode must NOT approximate the original dense weights \
         (err {dense_err:.3} vs scale {dense_scale:.3}); if it does, the buffer is not sparse"
    );

    // Round trip through the byte format is stable.
    let repacked = dequant_grey_raven(&pack_grey_raven(&s), n).expect("repack");
    assert_eq!(repacked, decoded, "pack/unpack must be idempotent");
}

fn round_e4m3(v: f32) -> f32 {
    grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(v))
}

/// E5 — malformed buffers are rejected loudly.
///
/// A decoder that clamps an out-of-range metadata code does not crash; it
/// returns a *different model* with no indication anything was wrong. That is
/// worse than a refusal, so every malformed shape gets an error and no value.
#[test]
fn e5_malformed_buffers_are_rejected_loudly() {
    let n = 64usize;
    let dense: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let s = sparsify_2_4_flat_with_fisher(&dense, &vec![1.0; n]).expect("multiple of 4");
    let good = pack_grey_raven(&s);
    assert!(dequant_grey_raven(&good, n).is_ok(), "control must decode");

    // Truncated buffer.
    for cut in [0usize, 1, good.len() / 2, good.len() - 1] {
        assert!(
            dequant_grey_raven(&good[..cut], n).is_err(),
            "truncated buffer ({cut} of {} bytes) must be rejected",
            good.len()
        );
    }
    // Over-long buffer.
    let mut long = good.clone();
    long.push(0);
    assert!(dequant_grey_raven(&long, n).is_err(), "over-long buffer must be rejected");

    // num_values not a multiple of 4.
    for bad_n in [1usize, 2, 3, 5, 63] {
        assert!(
            dequant_grey_raven(&good, bad_n).is_err(),
            "num_values={bad_n} must be rejected rather than padded"
        );
    }
    // num_values inconsistent with a buffer sized for another shape.
    assert!(
        dequant_grey_raven(&good, n * 2).is_err(),
        "a buffer sized for {n} must not decode as {n} values"
    );

    // Metadata code 6 and 7 name no valid slot pair (only 0..=5 exist). These
    // must error, not clamp to 0 and silently reconstruct a different model.
    for bad_code in [6u32, 7] {
        let mut buf = good.clone();
        let meta_start = n / 4 * 2;
        // Code lives in 3 bits at metadata bit 0 for group 0.
        for k in 0..3 {
            let bit = k;
            let byte = meta_start + bit / 8;
            if bad_code & (1 << k) != 0 {
                buf[byte] |= 1 << (bit % 8);
            } else {
                buf[byte] &= !(1 << (bit % 8));
            }
        }
        let err = dequant_grey_raven(&buf, n);
        assert!(
            err.is_err(),
            "metadata code {bad_code} names no valid pair and must be rejected, got {:?}",
            err.ok()
        );
    }
}
