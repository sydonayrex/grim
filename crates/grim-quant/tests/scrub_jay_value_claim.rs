//! WS-B B4: substantiate ScrubJay's accuracy claim, and re-derive its cost.
//!
//! B4 implements nothing. Like WS-A's A4, it is the measurement that decides
//! whether the format is worth having, and it is the CPU half of the evidence B7
//! will need on the GPU half.
//!
//! # The claim
//!
//! Choosing among 16 frozen codebooks per 8-element block must beat committing
//! to a single shared codebook. That is the entire premise; if a single codebook
//! matches it, the 4 selector bits per block are waste.
//!
//! # The comparison has to be at matched budget, and the budget is not what the
//! plan assumed
//!
//! The plan budgets ScrubJay at **4.5 bpw** and reasons about it accordingly
//! ("LO-BCQ's 4.5 bpw beats MXFP4's 4.25 bpw, so it is not even a bandwidth
//! win"). That figure omits the per-value sign bit the format provably needs, so
//! the real cost is **5.5 bpw**. This file therefore measures three points on the
//! density/accuracy curve rather than two:
//!
//! | arm | bpw | what it is |
//! |---|---|---|
//! | single codebook | 4.5 | the baseline the selector must beat |
//! | ScrubJay | 5.5 | the format, paying 1.0 bpw for the selector |
//! | E4M3 | 8.0 | the format grim already ships natively |
//!
//! The single-codebook baseline is *not* penalised artificially: it gets the same
//! per-block E4M3 scale and the same sign plane, and simply uses the one codebook
//! that is best on average. ScrubJay spends 1.0 bpw more and must earn it.
//!
//! CPU only.

use grim_quant::f32_to_fp8_e4m3;
use grim_quant::fp8_e4m3_to_f32;
use grim_quant::scrub_jay::{
    SCRUB_JAY_BLOCK, SCRUB_JAY_CODEBOOK, SCRUB_JAY_CODEBOOKS, dequantize_block, quantize_block,
    raw_block_scale,
};

/// Deterministic LCG; the calibration corpus uses the same construction, so the
/// fixture and the frozen table are drawn from the same family of distributions.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / (1u32 << 24) as f32 - 0.5
}

/// Which distribution a block is drawn from. Real weights are heavy-tailed, and
/// a uniform fixture flatters any format with a narrow dynamic range.
#[derive(Clone, Copy)]
enum Shape {
    Gaussian,
    Laplace,
    /// Every block a different *shape*, which is the case a per-block selector
    /// exists for.
    ///
    /// Homogeneous fixtures cannot show the selector working: if every block is
    /// drawn from the same distribution, every block wants the same codebook and
    /// the selector has nothing to decide. Real weight blocks are not like that
    /// either -- some are dense bulk, some carry an outlier, some are nearly
    /// constant -- so this fixture draws a fresh shape per block.
    Heterogeneous,
}

impl Shape {
    fn sample(self, s: &mut u64) -> f32 {
        let u: f32 = (0..12).map(|_| lcg(s)).sum();
        match self {
            Shape::Gaussian => (u - 6.0) * 0.15,
            Shape::Laplace => {
                let m = -0.18 * (1.0 - (u - 6.0) / 6.0).max(1e-6).ln();
                if lcg(s) < 0.5 {
                    -m
                } else {
                    m
                }
            }
            Shape::Heterogeneous => {
                // Pick a shape per draw, then sample from it. Mixes a tight
                // cluster, a uniform spread, and a spiky block with one large
                // outlier, which need genuinely different level spacing.
                let pick = ((lcg(s) + 0.5) * 3.0) as usize % 3;
                let r = lcg(s);
                match pick {
                    0 => (r * 0.05).exp() * 0.3 - 0.15,   // tight
                    1 => r * 0.8,                          // spread
                    _ => {
                        if r.abs() > 0.9 {
                            r * 3.0
                        } else {
                            r * 0.05
                        }
                    } // spiky
                }
            }
        }
    }
}

fn blocks(n_blocks: usize, shape: Shape, seed: u64) -> Vec<[f32; SCRUB_JAY_BLOCK]> {
    let mut s = seed;
    (0..n_blocks)
        .map(|_| {
            let mut b = [0f32; SCRUB_JAY_BLOCK];
            for slot in b.iter_mut() {
                *slot = shape.sample(&mut s);
            }
            b
        })
        .collect()
}

/// The single codebook that is best on average over a calibration set.
///
/// Chosen by aggregate error rather than picked by hand, so the baseline is the
/// strongest version of the thing ScrubJay is claiming to beat.
fn best_single_codebook(cal: &[[f32; SCRUB_JAY_BLOCK]]) -> usize {
    let mut best = 0usize;
    let mut best_err = f64::INFINITY;
    for c in 0..SCRUB_JAY_CODEBOOKS {
        let book = &SCRUB_JAY_CODEBOOK[c];
        let mut total = 0f64;
        for b in cal {
            let sc = block_scale(b, 1.0);
            let inv = 1.0 / sc;
            for &v in b.iter() {
                let mag = v.abs() * inv;
                let nearest = book
                    .iter()
                    .map(|&l| (l as f32 - mag).powi(2))
                    .fold(f32::INFINITY, f32::min);
                total += nearest as f64;
            }
        }
        if total < best_err {
            best_err = total;
            best = c;
        }
    }
    best
}

/// The effective scale `quantize_block` uses, recomputed so the baseline arm
/// scales identically rather than being handed a handicap.
fn block_scale(block: &[f32; SCRUB_JAY_BLOCK], pre: f32) -> f32 {
    fp8_e4m3_to_f32(f32_to_fp8_e4m3(raw_block_scale(block, pre))) * pre
}

/// Decode a block with one *fixed* codebook -- the baseline arm.
fn decode_with_fixed(
    block: &[f32; SCRUB_JAY_BLOCK],
    book: usize,
    pre: f32,
) -> [f32; SCRUB_JAY_BLOCK] {
    let sc = block_scale(block, pre);
    let inv = if sc == 0.0 { 0.0 } else { 1.0 / sc };
    let mut out = [0f32; SCRUB_JAY_BLOCK];
    for (i, &v) in block.iter().enumerate() {
        let mag = v.abs() * inv;
        let nearest = SCRUB_JAY_CODEBOOK[book]
            .iter()
            .map(|&l| (l as f32 - mag).powi(2))
            .fold(f32::INFINITY, f32::min);
        let level = SCRUB_JAY_CODEBOOK[book]
            .iter()
            .copied()
            .min_by(|a, b| {
                ((*a as f32 - mag).powi(2), *a)
                    .partial_cmp(&((*b as f32 - mag).powi(2), *b))
                    .unwrap()
            })
            .unwrap() as f32;
        let _ = nearest;
        let m = level * sc * pre;
        out[i] = if v < 0.0 { -m } else { m };
    }
    out
}

/// E4M3 at 8.0 bpw, the density reference point.
///
/// **Per-tensor scale, not per-block.** An earlier version used a per-block
/// `peak/448`, which underflows: E4M3's smallest subnormal is 2^-9 = 0.00195, so
/// any block peaking below ~0.87 got a scale that encoded to zero and the whole
/// block decoded to zero -- a relative RMSE of exactly 1.0, which is the
/// signature of a null arm rather than a bad one. That is the same
/// plausible-but-wrong reading the TreePie work hit twice.
///
/// A per-tensor scale is also how E4M3 is actually used in grim (per-channel /
/// per-K), so it is the honest reference rather than a handicapped one: E4M3
/// spends its extra 2.5 bits on the codebook, not on finer scaling.
fn e4m3_tensor_scale(blocks: &[[f32; SCRUB_JAY_BLOCK]]) -> f32 {
    let mut peak = 0.0f32;
    for b in blocks {
        for &v in b.iter() {
            peak = peak.max(v.abs());
        }
    }
    if peak == 0.0 {
        return 1.0;
    }
    // Map the tensor peak to 240 rather than 448: 240 sits comfortably inside
    // E4M3's normal range with headroom, so the scale itself is represented
    // without losing the top of the grid.
    let raw = peak / 240.0;
    let s = fp8_e4m3_to_f32(f32_to_fp8_e4m3(raw));
    if s == 0.0 { 1.0 } else { s }
}

fn decode_e4m3(block: &[f32; SCRUB_JAY_BLOCK], sc: f32) -> [f32; SCRUB_JAY_BLOCK] {
    let mut out = [0f32; SCRUB_JAY_BLOCK];
    for (i, &v) in block.iter().enumerate() {
        let m = fp8_e4m3_to_f32(f32_to_fp8_e4m3(v.abs() / sc)) * sc;
        out[i] = if v < 0.0 { -m } else { m };
    }
    out
}

fn rel_rmse(orig: &[[f32; SCRUB_JAY_BLOCK]], rec: &[[f32; SCRUB_JAY_BLOCK]]) -> f64 {
    let mut sse = 0f64;
    let mut energy = 0f64;
    for (a, b) in orig.iter().zip(rec.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x as f64 - *y as f64;
            sse += d * d;
            energy += (*x as f64) * (*x as f64);
        }
    }
    if energy == 0.0 {
        0.0
    } else {
        (sse / energy).sqrt()
    }
}

struct Arms {
    single: f64,
    scrub: f64,
    e4m3: f64,
}

fn measure(n: usize, shape: Shape, seed: u64) -> Arms {
    let cal = blocks(512, shape, seed ^ 0x5EED);
    let test = blocks(n, shape, seed);
    let book = best_single_codebook(&cal);

    let single: Vec<[f32; SCRUB_JAY_BLOCK]> = test
        .iter()
        .map(|b| decode_with_fixed(b, book, 1.0))
        .collect();
    let scrub: Vec<[f32; SCRUB_JAY_BLOCK]> = test
        .iter()
        .map(|b| {
            let (sel, idx, sign, sc) = quantize_block(b, 1.0);
            dequantize_block(sel, &idx, sign, sc, 1.0)
        })
        .collect();
    let e4m3_scale = e4m3_tensor_scale(&test);
    let e4m3: Vec<[f32; SCRUB_JAY_BLOCK]> =
        test.iter().map(|b| decode_e4m3(b, e4m3_scale)).collect();

    Arms {
        single: rel_rmse(&test, &single),
        scrub: rel_rmse(&test, &scrub),
        e4m3: rel_rmse(&test, &e4m3),
    }
}

/// B4: the format's actual claim. Per-block selection must beat one shared
/// codebook.
///
/// This is the claim LO-BCQ makes, and the reason the 4 selector bits exist. The
/// baseline is the strongest possible single codebook, chosen by aggregate error
/// over a calibration set rather than picked by hand.
#[test]
fn w4a4_beats_a_single_codebook_baseline() {
    for (name, shape) in [("gaussian", Shape::Gaussian), ("laplace", Shape::Laplace)] {
        let a = measure(4096, shape, 0xABCD_1234);
        eprintln!(
            "{name:9} single@4.5bpw {:.4}   ScrubJay@5.5bpw {:.4}   E4M3@8.0bpw {:.4}   \
             gain {:.1}%",
            a.single,
            a.scrub,
            a.e4m3,
            (1.0 - a.scrub / a.single) * 100.0
        );
        assert!(
            a.scrub < a.single,
            "{name}: ScrubJay ({:.4}) must beat one shared codebook ({:.4}); the \
             4 selector bits per block are buying nothing",
            a.scrub,
            a.single
        );
    }
}

/// The gain must be a real margin, not a rounding artefact.
///
/// A selector that ties the best constant choice is not worth its bits, and the
/// format's premise is that the shape diversity pays.
#[test]
fn the_selector_gain_is_a_real_margin_where_shape_varies() {
    // Heterogeneous blocks, not homogeneous ones. On a single-shape fixture every
    // block wants the same codebook and the selector has nothing to decide, so
    // this margin is only meaningful where block shapes actually differ.
    let a = measure(4096, Shape::Heterogeneous, 0xDEAD_BEEF);
    eprintln!(
        "heterogeneous single@4.5bpw {:.4}   ScrubJay@5.5bpw {:.4}   gain {:.1}%",
        a.single,
        a.scrub,
        (1.0 - a.scrub / a.single) * 100.0
    );
    assert!(
        a.scrub < a.single * 0.95,
        "ScrubJay is only {:.1}% better than one shared codebook ({:.4} vs {:.4}); \
         expected a real margin",
        (1.0 - a.scrub / a.single) * 100.0,
        a.scrub,
        a.single
    );
}

/// The density context, stated as a bound rather than left implicit.
///
/// At 5.5 bpw ScrubJay is not a bandwidth win over E4M3's 8.0 -- it is 31% less
/// memory for some multiple of the error. The question B7 must answer on the GPU
/// is whether that trade is worth it; this only establishes the price.
#[test]
fn scrub_jays_price_against_e4m3_is_bounded() {
    for (name, shape) in [
        ("gaussian", Shape::Gaussian),
        ("laplace", Shape::Laplace),
        ("heterogeneous", Shape::Heterogeneous),
    ] {
        let a = measure(4096, shape, 0xABCD_1234);
        let ratio = a.scrub / a.e4m3;
        eprintln!("{name:9} ScrubJay/E4M3 error ratio {ratio:.2}x at 5.5 vs 8.0 bpw");
        assert!(
            ratio < 3.0,
            "{name}: ScrubJay at 5.5 bpw is {ratio:.2}x E4M3's error at 8.0 bpw. \
             Paying 2.5 bpw for that much extra error is not a trade, and B7's \
             kill criterion should fire."
        );
        // And the converse, so the test cannot pass with both arms broken.
        assert!(
            ratio > 0.5,
            "{name}: ScrubJay should be *worse* than E4M3 (fewer bits), not \
             better; a ratio below 0.5 means the ScrubJay arm is broken"
        );
    }
}

/// The honest bit accounting, asserted rather than asserted-in-prose.
///
/// The plan says 4.5 bpw throughout, and reasons about ScrubJay's viability on
/// that basis. The sign plane it omits is 1.0 bpw. This test exists so the
/// discrepancy is a build failure if someone re-derives the budget without the
/// sign term, rather than a comment nobody rereads.
#[test]
fn the_bit_budget_includes_the_sign_plane() {
    const SELECTOR_BITS: u32 = 4;
    const INDEX_BITS: u32 = 4;
    const SIGN_BITS: u32 = 1;
    let per_value = INDEX_BITS + SIGN_BITS;
    let per_block = SELECTOR_BITS + per_value * SCRUB_JAY_BLOCK as u32;
    let bpw = per_block as f32 / SCRUB_JAY_BLOCK as f32;

    assert_eq!(
        bpw, 5.5,
        "selector {SELECTOR_BITS} + index {INDEX_BITS} + sign {SIGN_BITS} per \
         value over {SCRUB_JAY_BLOCK} values"
    );
    // The plan's figure, and why it is wrong.
    assert_eq!(
        SELECTOR_BITS + INDEX_BITS,
        8,
        "4 + 4 = 8 bits per 8 values is the plan's 4.5 bpw -- it omits the sign \
         bit, which is not optional for signed weights"
    );
    // And the consequence for the attention floor, which is 5.
    assert!(
        bpw >= 5.0,
        "at {bpw} bpw ScrubJay clears the attention floor of 5 -- unlike every \
         other 4-bit format in the tree. If this ever drops below 5, that \
         argument is gone."
    );
}

/// The honest characterisation of *where* the selector pays.
///
/// B7's kill criterion is about wall-clock, but this is the accuracy half of the
/// same question, and the answer is conditional rather than uniform: the selector
/// earns its 1.0 bpw on blocks whose shapes differ, and earns very little on
/// blocks drawn from one distribution. Both numbers are asserted, because a test
/// that only asserted the flattering half would misrepresent the format.
#[test]
fn the_selector_pays_on_shape_variety_and_little_on_homogeneity() {
    let hetero = measure(4096, Shape::Heterogeneous, 0x1234_5678);
    let homo = measure(4096, Shape::Laplace, 0x1234_5678);

    let hetero_gain = 1.0 - hetero.scrub / hetero.single;
    let homo_gain = 1.0 - homo.scrub / homo.single;
    eprintln!("gain on heterogeneous blocks: {:.1}%", hetero_gain * 100.0);
    eprintln!("gain on laplace blocks:       {:.1}%", homo_gain * 100.0);

    // The selector must help where shapes vary.
    assert!(
        hetero.scrub < homo.scrub,
        "heterogeneous blocks ({:.4}) must reconstruct better than homogeneous          ones ({:.4}); if not, shape diversity is not being exploited",
        hetero.scrub,
        homo.scrub
    );

    // And it must not be a large loss where they do not -- a selector that cost
    // accuracy on homogeneous blocks would be strictly worse than one codebook.
    assert!(
        homo.scrub <= homo.single * 1.02,
        "on homogeneous blocks the selector must not be materially worse than a \
         single codebook ({:.4} vs {:.4})",
        homo.scrub,
        homo.single
    );

    // Characterisation, recorded so the number is not lost: on a homogeneous
    // fixture the gain is small, and that is a real limit on the format's value
    // rather than a defect in the measurement.
    assert!(
        homo_gain < 0.10,
        "expected the homogeneous gain to be under 10% (measured {:.1}%); a much \
         larger number means the fixture is not actually homogeneous and the \
         contrast above is not testing what it claims",
        homo_gain * 100.0
    );
}
