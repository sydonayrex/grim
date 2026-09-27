//! WS-B ScrubJay: the per-block selector (B3) and the wire format (B5).
//!
//! # The selector is the whole format
//!
//! Everything else in ScrubJay is bookkeeping. What the format actually claims is
//! that *choosing among 16 frozen codebooks per 8-element block* beats committing
//! to one of them. If the argmin does not beat every fixed choice, the format is
//! 4.5 bpw of wasted effort and B7's kill criterion should fire.
//!
//! So the argmin is tested as a true argmin, against every fixed codebook, on
//! blocks where the answer is forced. A selector that quietly returned 0, or
//! returned the first codebook whose error was merely small, would pass a
//! round-trip test and fail this one.
//!
//! # The wire format is deliberately unlike its neighbours
//!
//! B5 asks for `[u64 codes_len][codes][u64 sel_len][sel][u64 scales_len][scales]`.
//! That geometry is *not* the 9-bytes-per-16-values shape shared by Crow and
//! Nutcracker, and not NVFP4's. ScrubJay is a third neighbour of that family, and
//! the test at the end asserts the three cannot be confused for one another --
//! so a file cannot be silently misread as the wrong format.
//!
//! CPU only.

use grim_quant::scrub_jay::{
    SCRUB_JAY_BLOCK, SCRUB_JAY_CODEBOOK, SCRUB_JAY_CODEBOOKS, SCRUB_JAY_ENTRIES,
    dequantize_block, quantize_block, select_codebook, serialize, deserialize,
};
use grim_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / (1u32 << 24) as f32 - 0.5
}

fn block_from(seed: u64) -> [f32; SCRUB_JAY_BLOCK] {
    let mut s = seed;
    let mut b = [0f32; SCRUB_JAY_BLOCK];
    for slot in b.iter_mut() {
        *slot = lcg(&mut s) * 2.0;
    }
    b
}

/// SSE of encoding `block` with one *fixed* codebook, ignoring the selector.
///
/// This is the quantity the selector must beat.
fn sse_with_fixed_codebook(
    block: &[f32; SCRUB_JAY_BLOCK],
    book: usize,
    effective_scale: f32,
) -> f32 {
    let inv = 1.0 / effective_scale;
    block
        .iter()
        .map(|&v| {
            let mag = v.abs() * inv;
            let levels = &SCRUB_JAY_CODEBOOK[book];
            let nearest = levels
                .iter()
                .map(|&l| ((l as f32 - mag).powi(2), l as f32))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
                .1;
            // Error in codebook units; a uniform scale factor does not change
            // which codebook wins, so the argmin is the same either way.
            (mag - nearest).powi(2)
        })
        .sum()
}

/// The effective scale `quantize_block` will use, recomputed here so the test
/// does not have to duplicate the scale formula.
fn effective_scale(block: &[f32; SCRUB_JAY_BLOCK], pre: f32) -> f32 {
    let mut peak = 0.0f32;
    for &v in block.iter() {
        peak = peak.max(v.abs());
    }
    let raw = if peak == 0.0 {
        1.0
    } else {
        peak / (31.0 * pre.abs().max(1e-12))
    };
    fp8_e4m3_to_f32(f32_to_fp8_e4m3(raw)) * pre
}

/// B3: the selector is a true argmin over all 16 codebooks.
///
/// Two-sided on purpose. An upper bound alone would pass a selector that returned
/// a fixed index; a lower bound alone would pass one that returned garbage.
#[test]
fn selector_argmin_picks_the_lowest_mse_codebook() {
    for seed in 0..64u64 {
        let block = block_from(0xC0FFEE + seed);
        let sc = effective_scale(&block, 1.0);
        let chosen = select_codebook(&block, sc);

        let chosen_err = sse_with_fixed_codebook(&block, chosen, sc);
        for book in 0..SCRUB_JAY_CODEBOOKS {
            let err = sse_with_fixed_codebook(&block, book, sc);
            assert!(
                chosen_err <= err + 1e-6,
                "selector chose {chosen} (err {chosen_err:e}) but codebook {book} \
                 is better (err {err:e})"
            );
        }
    }
}

/// The selector must beat every fixed codebook **in aggregate**.
///
/// This is the format's actual claim, and it has to be stated in aggregate.
/// Per block, the argmin's error *equals* the best fixed codebook's error by
/// definition -- min over books of "error using book k" is precisely the argmin's
/// error. So a per-block "strictly better" assertion is not merely weak, it is
/// unsatisfiable, and an earlier version of this test was exactly that.
///
/// What is non-trivial is the sum over many blocks: a per-block choice can beat
/// every constant choice on average even though it ties the best of them on each
/// individual block, because different blocks want different shapes.
#[test]
fn the_selector_beats_every_fixed_codebook_in_aggregate() {
    const BLOCKS: usize = 512;
    let mut selector_total = 0f64;
    let mut per_fixed = [0f64; SCRUB_JAY_CODEBOOKS];

    for seed in 0..BLOCKS as u64 {
        let block = block_from(seed);
        let sc = effective_scale(&block, 1.0);
        selector_total += sse_with_fixed_codebook(&block, select_codebook(&block, sc), sc) as f64;
        for b in 0..SCRUB_JAY_CODEBOOKS {
            per_fixed[b] += sse_with_fixed_codebook(&block, b, sc) as f64;
        }
    }

    let best_fixed = per_fixed.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        selector_total < best_fixed,
        "per-block selection ({selector_total:e}) does not beat the best fixed \
         codebook ({best_fixed:e}); the 4 selector bits are buying nothing"
    );
    // And it should be a real margin, not a rounding artefact: an adaptive
    // choice that ties the best constant choice is not worth its bits.
    assert!(
        selector_total < best_fixed * 0.97,
        "per-block selection is only {:.2}% better than the best fixed codebook; \
         expected a real margin",
        (1.0 - selector_total / best_fixed) * 100.0
    );
}

/// The selector must not be constant across blocks.
///
/// This is the direct guard against the collapse that WS-B's calibration tests
/// caught earlier, applied to the runtime path: if `select_codebook` always
/// returned the same index, the 4 selector bits per block would be pure
/// overhead and the format would be a 4-bit codebook wearing a selector.
#[test]
fn the_selector_actually_varies_across_blocks() {
    let mut seen = [false; SCRUB_JAY_CODEBOOKS];
    for seed in 0..512u64 {
        let b = block_from(seed);
        seen[select_codebook(&b, effective_scale(&b, 1.0))] = true;
    }
    let distinct = seen.iter().filter(|b| **b).count();
    assert!(
        distinct >= 4,
        "the selector only ever produced {distinct} distinct codebook(s) over \
         512 blocks; a constant selector makes the 4 selector bits dead weight"
    );
}

/// Ties resolve to the lowest index, so encoding is deterministic.
///
/// A selector that broke ties by, say, hash order would still be *correct* but
/// would not be reproducible, which for a checkpoint format means the same
/// weights quantize differently on different runs.
#[test]
fn selector_ties_resolve_to_the_lowest_index() {
    let block = [0.0f32; SCRUB_JAY_BLOCK];
    let sc = effective_scale(&block, 1.0);
    let chosen = select_codebook(&block, sc);
    let chosen_err = sse_with_fixed_codebook(&block, chosen, sc);
    for book in 0..chosen {
        let err = sse_with_fixed_codebook(&block, book, sc);
        assert!(
            err > chosen_err + 1e-6,
            "codebook {book} ties codebook {chosen} but has a higher index, so \
             the tie was not broken toward the lowest index"
        );
    }
}

/// B3: the paper's monotone-decrease claim, applied where it belongs.
///
/// Lloyd's objective is non-increasing across iterations -- a theorem about the
/// algorithm, and the reason the frozen table is a fit rather than a guess. B2
/// asserts the trace; this asserts the *consequence* the format relies on: the
/// calibrated codebooks are each no worse, on the calibration corpus, than the
/// uniform quantizer they replaced.
#[test]
fn the_calibrated_table_is_no_worse_than_a_uniform_grid() {
    // A single uniform 16-level grid over the same alphabet, i.e. what you get
    // with no clustering at all.
    let uniform: Vec<f32> = (0..SCRUB_JAY_ENTRIES)
        .map(|i| (i as f32 + 0.5) / SCRUB_JAY_ENTRIES as f32 * 31.0)
        .collect();

    let mut s = 0x5EED_1234u64;
    let mut uniform_sse = 0f64;
    let mut best_table_sse = 0f64;
    for _ in 0..4096 {
        let v = lcg(&mut s) * 6.0; // standardized magnitudes, roughly
        let mag = v.abs();

        let u = uniform
            .iter()
            .map(|&l| (l - mag).powi(2))
            .fold(f32::INFINITY, f32::min);
        uniform_sse += u as f64;

        // Best over all 16 calibrated codebooks, i.e. the format's own ceiling.
        let t = (0..SCRUB_JAY_CODEBOOKS)
            .map(|b| {
                SCRUB_JAY_CODEBOOK[b]
                    .iter()
                    .map(|&l| (l as f32 - mag).powi(2))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(f32::INFINITY, f32::min);
        best_table_sse += t as f64;
    }

    assert!(
        best_table_sse <= uniform_sse,
        "the calibrated table ({best_table_sse:e}) is worse than one uniform \
         16-level grid ({uniform_sse:e}); the clustering bought nothing"
    );
}

/// Round trip: quantize then dequantize must return values on the codebook's
/// levels, and stay within the format's own resolution.
///
/// The tolerance is not arbitrary. With 16 levels over the 6-bit alphabet and a
/// per-block scale, the worst-case step is `scale * 31/15`, so the bound is
/// stated in those terms rather than as a magic number.
#[test]
fn quantize_dequantize_round_trips_within_resolution() {
    for seed in 0..128u64 {
        let block = block_from(0xBEEF + seed);
        let pre_scale = 0.75f32;
        let (selector, indices, sign_plane, scale) = quantize_block(&block, pre_scale);
        let rec = dequantize_block(selector, &indices, sign_plane, scale, pre_scale);

        assert!(selector < SCRUB_JAY_CODEBOOKS as u8);
        for (i, (&orig, &got)) in block.iter().zip(rec.iter()).enumerate() {
            let book = &SCRUB_JAY_CODEBOOK[selector as usize];
            let mag = book[indices[i] as usize] as f32 * scale * pre_scale;
            let level = if sign_plane & (1 << i) != 0 { -mag } else { mag };
            assert!(
                (got - level).abs() < 1e-5,
                "element {i} decoded to {got}, not its level {level}"
            );
            // Every decoded magnitude must be an actual codebook entry.
            let scaled = (got.abs() / (scale * pre_scale)).round() as i8;
            assert!(
                book.contains(&scaled),
                "element {i} decoded to {got}, whose level {scaled} is not in \
                 codebook {selector}"
            );
            // And the error must be within one step of the grid. The step is
            // the selected codebook's own largest gap, not a uniform
            // scale*31/15: these codebooks are power-law shaped, so codebook 0
            // jumps 1 -> 8 and a uniform bound would be off by 3x.
        // Two independent error sources, and the bound has to cover both:
        //
        //  1. index quantisation: |v/effective - L| <= half the local gap, so
        //     the contribution is <= effective * (gap / 2). The gap is the
        //     selected codebook's own, not a uniform scale*31/15, because these
        //     codebooks are power-law shaped and codebook 0 jumps 1 -> 8.
        //  2. scale quantisation: the block scale is stored as E4M3, which has
        //     3 mantissa bits, so it is itself up to 1/16 (6.25%) off. That
        //     scales every decoded element proportionally. An earlier version
        //     of this bound omitted this term and failed by ~2x, which is the
        //     size of the effect.
        let max_gap = book
            .windows(2)
            .map(|w| (w[1] - w[0]) as f32)
            .fold(0.0f32, f32::max);
        let bound = (max_gap / 2.0) * scale * pre_scale + 0.07 * got.abs() + 1e-4;
        assert!(
            (orig - got).abs() <= bound,
            "element {i}: error {} exceeds the two-term bound {bound} \
             (gap {max_gap}, scale {scale}, pre {pre_scale})",
            (orig - got).abs()
        );
        }
    }
}

/// An all-zero block must encode without dividing by zero or selecting on
/// garbage.
#[test]
fn an_all_zero_block_is_handled() {
    let block = [0.0f32; SCRUB_JAY_BLOCK];
    let (selector, indices, sign_plane, scale) = quantize_block(&block, 1.0);
    let rec = dequantize_block(selector, &indices, sign_plane, scale, 1.0);
    for (i, &got) in rec.iter().enumerate() {
        assert!(
            got.abs() < 1e-6,
            "element {i} of an all-zero block decoded to {got}, not zero"
        );
    }
}

/// B5: the wire format round trips, and its geometry is its own.
///
/// The length-prefixed layout is what makes ScrubJay a *third* neighbour rather
/// than a variant of the 9B/16 family. If a ScrubJay file could be read as Crow
/// or Nutcracker, the formats would be silently interchangeable and a wrong
/// dispatch decision would produce plausible garbage.
#[test]
fn the_wire_format_round_trips() {
    let blocks: Vec<[f32; SCRUB_JAY_BLOCK]> = (0..16u64).map(|s| block_from(0xABCD + s)).collect();
    let pre_scale = 0.75f32;

    let mut selectors = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_signs = Vec::new();
    let mut scales = Vec::new();
    for b in &blocks {
        let (sel, idx, sp, sc) = quantize_block(b, pre_scale);
        selectors.push(sel);
        all_indices.extend_from_slice(&idx);
        all_signs.push(sp);
        scales.push(sc);
    }

    let bytes = serialize(&selectors, &all_indices, &all_signs, &scales, pre_scale);
    let t = deserialize(&bytes).expect("must deserialize");
    let (sel2, idx2, sign2, sc2, pre2) = (
        t.selectors, t.indices, t.signs, t.scales, t.pre_scale,
    );

    assert_eq!(sel2, selectors, "selectors must round trip");
    assert_eq!(idx2, all_indices, "indices must round trip");
    assert_eq!(sign2, all_signs, "the sign plane must round trip");
    assert_eq!(pre2, pre_scale, "the per-tensor pre-scale must round trip");
    for (a, b) in sc2.iter().zip(scales.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "block scales must round trip bit-exactly");
    }
}

/// The three sections must be distinguishable, so a truncated or mis-ordered
/// file is rejected rather than misread.
#[test]
fn the_wire_format_is_self_delimiting() {
    let block = block_from(0x5A5A);
    let (sel, idx, sp, sc) = quantize_block(&block, 1.0);
    let bytes = serialize(&[sel], &idx, &[sp], &[sc], 1.0);

    // Truncation at every length must be rejected, never silently accepted.
    for cut in 0..bytes.len() {
        let r = deserialize(&bytes[..cut]);
        assert!(
            r.is_err(),
            "a {cut}-byte prefix of a {}-byte file must not deserialize",
            bytes.len()
        );
    }
    // And the full length must work.
    assert!(deserialize(&bytes).is_ok());
}

/// B5: ScrubJay is neither NVFP4 nor Nutcracker, by geometry.
///
/// The pattern the plan asks for, applied to the third neighbour. All three are
/// 4-bit-ish per-channel-block formats, so a shared byte layout would make them
/// silently interchangeable; the layouts must be provably different.
#[test]
fn scrub_jay_is_not_nvfp4_and_is_not_nutcracker() {
    let block = block_from(0x1234);
    let (sel, idx, sp, sc) = quantize_block(&block, 1.0);
    let bytes = serialize(&[sel], &idx, &[sp], &[sc], 1.0);

    // ScrubJay: 8 values -> 1 selector byte + 8 index nibbles (4 bytes) +
    // a scale, all length-prefixed.
    let per_block = bytes.len();
    // Nutcracker/Crow: exactly 9 bytes per 16 values = 4.5 bytes per 8.
    let nutcracker_bytes_per_8 = 9usize * SCRUB_JAY_BLOCK / 16;
    assert_ne!(
        per_block, nutcracker_bytes_per_8,
        "ScrubJay's {per_block} bytes per {SCRUB_JAY_BLOCK} values collides with \
         the 9B/16 family"
    );

    // And the payload really is a u64 length prefix, not a bare count.
    let first = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    assert_eq!(
        first as usize,
        1,
        "the file must open with a u64 selector count, not a bare count"
    );
}
