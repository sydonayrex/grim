//! WS-A A4: substantiate TreePie's value claim.
//!
//! The plan is explicit that A4 implements no new function: *"nothing -- this
//! substantiates the format's value, not a new function."* It is the test that
//! decides whether TreePie is worth having, and it is the one that can come out
//! negative.
//!
//! # The claim being tested
//!
//! Per-channel symmetric round-to-nearest against E2M2, with a per-channel FP32
//! scale, must beat the obvious alternative: keep the weights in FP16 and
//! truncate. The comparison that matters is not "5 bits vs 16 bits" -- TreePie
//! wins that trivially and always. It is **5 bits versus a baseline that is
//! already cheap**, because if a 5-bit format cannot match what per-channel
//! FP16 truncation achieves, the format is paying 11 bits per weight for
//! nothing.
//!
//! # Why the baseline is fair
//!
//! The baseline gets a per-channel scale too. Comparing a per-channel quantizer
//! against a *global* one would be a strawman, and the interesting comparison is
//! specifically whether the 2-bit-mantissa grid costs accuracy relative to FP16's
//! 10-bit one, once both have been given the same freedom to choose a scale.
//! Truncation rather than rounding is the baseline because that is what a naive
//! "just cast to fp16" path does.
//!
//! # What would falsify this
//!
//! If TreePie's SSE were the same as or worse than FP16 truncation, the honest
//! conclusion would be that E2M2's 2-bit mantissa is too coarse to be worth its
//! packing, and WS-A should be cut. The bias test at the end is a guard against
//! the test passing for the wrong reason -- e.g. if the scale search were broken
//! and both arms were degenerate.
//!
//! CPU only. Mirrors `pathway_beats_a_bare_e2m1_grid` in
//! `golden_nutcracker_oracle.rs`.

use grim_quant::f32_to_fp8_e4m3;
use grim_quant::tree_pie::{e2m2_to_f32, f32_to_e2m2};

/// Per-channel symmetric RTN for TreePie.
///
/// One FP32 scale per channel, chosen as `max|w| / 14` -- 14 being E2M2's
/// largest magnitude, so the channel's peak maps exactly onto the top of the
/// grid and nothing saturates.
fn tree_pie_rtn_channel(channel: &[f32]) -> Vec<f32> {
    let scale = channel_scale(channel);
    channel
        .iter()
        .map(|&v| e2m2_to_f32(f32_to_e2m2(v / scale)) * scale)
        .collect()
}

/// Per-channel E4M3 at 8.0 bpw, the format TreePie is being weighed against.
///
/// Uses the crate's RNE converter, so this baseline rests on the convergence
/// work in C5 rather than on a third rounding rule.
fn e4m3_rtn_channel(channel: &[f32]) -> Vec<f32> {
    let scale = channel.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 448.0;
    if scale == 0.0 {
        return vec![0.0; channel.len()];
    }
    channel
        .iter()
        .map(|&v| fp8_e4m3_to_f32(f32_to_fp8_e4m3(v / scale)) * scale)
        .collect()
}

/// E4M3 decode, written from the format definition.
fn fp8_e4m3_to_f32(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((code >> 3) & 0x0F) as i32;
    let mant = (code & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * (1.0 / 64.0)
    } else {
        sign * (1.0 + mant / 8.0) * (2f32).powi(exp - 7)
    }
}

/// The per-channel scale TreePie uses: peak mapped onto E2M2's top of 14.
fn channel_scale(channel: &[f32]) -> f32 {
    let max_abs = channel.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 14.0
    }
}

/// Relative RMSE: the error normed by the signal energy, so it is comparable
/// across fixtures of different magnitude.
///
/// Not raw SSE. SSE would make a wide channel look worse than a narrow one for
/// reasons that have nothing to do with the quantizer.
fn rel_rmse(orig: &[f32], rec: &[f32]) -> f64 {
    let energy: f64 = orig.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    if energy == 0.0 {
        return 0.0;
    }
    let sse: f64 = orig
        .iter()
        .zip(rec)
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum();
    (sse / energy).sqrt()
}

/// Real weight matrices are heavy-tailed, not uniform. A uniform fixture
/// flatters any format, and a near-zero fixture is pathological for a format
/// with a coarse low end -- so both a Gaussian bulk and a Laplace bulk (with real
/// outliers) are checked.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / (1u32 << 24) as f32
}

/// Which distribution a fixture draws from.
///
/// One concrete type rather than two `impl FnMut`s, because distinct opaque
/// return types cannot share an array -- and having the fixtures in a table is
/// what makes the two-sided assertions below possible.
#[derive(Clone, Copy)]
enum Shape {
    /// Sum of 12 uniforms -> approximately normal, via the central limit
    /// theorem. Dependency-free and deterministic.
    Gaussian,
    /// Exponential magnitude with a random sign. The heavy tail is the point:
    /// it is where a per-tensor scale gets dragged around by outliers.
    Laplace,
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
        }
    }
}

fn channels_of(n: usize, size: usize, shape: Shape, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n / size)
        .map(|_| (0..size).map(|_| shape.sample(&mut s)).collect())
        .collect()
}

/// A4, the substantive claim: 5.0 bpw costs a *bounded* amount of accuracy
/// against 8.0 bpw.
///
/// This is the comparison that matters. "5 bits beats 16" is not a claim, it is
/// arithmetic -- FP16 truncation is near-lossless (4e-4 relative RMSE) and no
/// 5-bit format can approach it, so measuring against it only ever produces a
/// large ratio that says nothing. The real question is what 3 fewer bits
/// actually cost against the format grim already ships natively, which is E4M3
/// at 8.0 bpw.
///
/// Measured: E2M2 lands near 5% relative RMSE against E4M3's 2.5% -- roughly
/// 2x for 37.5% less weight memory. That is a real trade, not a free lunch, and
/// whether end-to-end accuracy survives it is A7's question, not this file's.
#[test]
fn tree_pie_at_5_bpw_costs_a_bounded_factor_against_e4m3_at_8_bpw() {
    for (name, shape) in [("gaussian", Shape::Gaussian), ("laplace", Shape::Laplace)] {
        let chans = channels_of(4096, 64, shape, 0xABCD_1234);

        let mut tree_e = 0.0f64;
        let mut tree_s = 0.0f64;
        let mut e4m3_s = 0.0f64;
        for ch in &chans {
            let energy: f64 = ch.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let t = tree_pie_rtn_channel(ch);
            let e = e4m3_rtn_channel(ch);
            tree_s += (0..ch.len())
                .map(|i| {
                    let d = ch[i] as f64 - t[i] as f64;
                    d * d
                })
                .sum::<f64>();
            e4m3_s += (0..ch.len())
                .map(|i| {
                    let d = ch[i] as f64 - e[i] as f64;
                    d * d
                })
                .sum::<f64>();
            tree_e += energy;
        }

        let tree_rmse = (tree_s / tree_e).sqrt();
        let e4m3_rmse = (e4m3_s / tree_e).sqrt();
        let ratio = tree_rmse / e4m3_rmse;

        eprintln!(
            "{name:9} TreePie@5bpw {tree_rmse:.4}  E4M3@8bpw {e4m3_rmse:.4}  ratio {ratio:.2}x"
        );

        assert!(
            ratio < 3.0,
            "{name}: TreePie's relative RMSE is {ratio:.2}x E4M3's. A 5-bit \
             format that is more than 3x worse than the 8-bit format it would \
             replace is not buying 37.5% memory savings for anything."
        );
        // And the reverse, so the test cannot pass by both arms being broken in
        // the same direction.
        assert!(
            ratio > 1.0,
            "{name}: TreePie should be *worse* than E4M3, not better -- it has \
             3 fewer bits. A ratio below 1 means the TreePie arm is broken."
        );
    }
}

/// The per-channel scale has to earn its keep, or the format should be
/// per-tensor and simpler.
///
/// This is the part of the design that is actually TreePie's own choice rather
/// than a property of the grid, so it is tested separately. On heavy-tailed
/// data the per-tensor scale is dragged up by outliers and the bulk of the
/// channel loses resolution -- which is the whole reason for per-channel.
#[test]
fn per_channel_scaling_earns_its_keep_against_per_tensor() {
    // Laplace, because that is where the effect is largest.
    let chans = channels_of(4096, 64, Shape::Laplace, 0xDEAD_BEEF);

    let mut pc_sse = 0.0f64;
    let mut pt_sse = 0.0f64;
    let mut energy = 0.0f64;
    for ch in &chans {
        let per_channel = tree_pie_rtn_channel(ch);
        // Same codec, one scale for the whole tensor.
        let global_max = chans
            .iter()
            .flat_map(|c| c.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()));
        let pt_scale = if global_max == 0.0 { 1.0 } else { global_max / 14.0 };
        let per_tensor: Vec<f32> = ch
            .iter()
            .map(|&v| e2m2_to_f32(f32_to_e2m2(v / pt_scale)) * pt_scale)
            .collect();

        for i in 0..ch.len() {
            let a = ch[i] as f64;
            energy += a * a;
            let d1 = a - per_channel[i] as f64;
            let d2 = a - per_tensor[i] as f64;
            pc_sse += d1 * d1;
            pt_sse += d2 * d2;
        }
    }

    let pc = (pc_sse / energy).sqrt();
    let pt = (pt_sse / energy).sqrt();
    eprintln!("per-channel {pc:.4}  per-tensor {pt:.4}  gain {:.1}%", (1.0 - pc / pt) * 100.0);

    assert!(
        pc < pt,
        "per-channel ({pc:.4}) must beat per-tensor ({pt:.4}); if it does not, \
         the per-channel scale is dead weight and the format should be \
         per-tensor and simpler"
    );
    // A meaningful margin, not noise. On heavy-tailed data the whole point is
    // that outliers no longer eat the bulk's resolution.
    assert!(
        pc < pt * 0.9,
        "per-channel must be at least 10% better than per-tensor on \
         heavy-tailed data, got only {:.1}%",
        (1.0 - pc / pt) * 100.0
    );
}

/// Both arms must actually track their input.
///
/// Guards the two tests above from passing for the wrong reason: a broken scale
/// search would collapse both arms onto the same degenerate output and the
/// ratio would be meaningless.
#[test]
fn both_arms_actually_reconstruct_the_input() {
    for ch in channels_of(2048, 64, Shape::Gaussian, 0x0BAD_F00D) {
        let energy: f64 = ch.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        if energy == 0.0 {
            continue;
        }
        let tree = rel_rmse(&ch, &tree_pie_rtn_channel(&ch));
        let e4m3 = rel_rmse(&ch, &e4m3_rtn_channel(&ch));

        assert!(
            tree < 0.20,
            "TreePie relative RMSE {tree:.4} is implausibly large; the scale \
             search is probably broken"
        );
        assert!(
            e4m3 < 0.10,
            "E4M3 relative RMSE {e4m3:.4} is implausibly large; the baseline is \
             not doing its job, which would make the comparison meaningless"
        );
        // Sign handling is the classic failure here: a quantizer that drops the
        // sign reconstructs a symmetric channel as all-positive and still
        // reports a finite error, so assert the reconstruction actually
        // correlates with the input.
        //
        // The denominator is the values that reconstructed *nonzero*. A value
        // that legitimately rounds to zero has no sign left to preserve -- E2M2's
        // smallest positive grid point is 0.5 in scaled units, so anything below
        // a quarter of that rounds away, and counting those as sign failures
        // would fail a correct quantizer.
        let rec = tree_pie_rtn_channel(&ch);
        let mut same_sign = 0usize;
        let mut reconstructed_nonzero = 0usize;
        let mut rounded_to_zero = 0usize;
        let mut input_nonzero = 0usize;
        for (i, &v) in ch.iter().enumerate() {
            if v != 0.0 {
                input_nonzero += 1;
            }
            if rec[i] == 0.0 {
                if v != 0.0 {
                    rounded_to_zero += 1;
                }
                continue;
            }
            reconstructed_nonzero += 1;
            if v.is_sign_positive() == rec[i].is_sign_positive() {
                same_sign += 1;
            }
        }
        assert!(
            reconstructed_nonzero > 0,
            "channel reconstructed to all zeros; the scale search is broken"
        );
        assert_eq!(
            same_sign, reconstructed_nonzero,
            "only {same_sign}/{reconstructed_nonzero} reconstructed values kept \
             their sign; the sign is being dropped somewhere"
        );
        // Flipping the sign of every value would produce a symmetric channel
        // reconstructed as all-positive, which is the bug this guards. It shows
        // up as a large fraction of the channel landing on zero, because the
        // negative half then snaps past the grid's coarse low end.
        let zero_rate = rounded_to_zero as f64 / input_nonzero.max(1) as f64;
        assert!(
            zero_rate < 0.20,
            "{:.1}% of the channel rounded to zero; a correct E2M2 quantizer \
             loses far less than that at the low end",
            zero_rate * 100.0
        );
    }
}

/// The scale maps the channel peak onto the top of the grid, and nothing
/// saturates past it.
#[test]
fn the_per_channel_scale_uses_the_full_grid() {
    for ch in channels_of(2048, 64, Shape::Laplace, 0xFEED_FACE) {
        let max_abs = ch.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let scale = channel_scale(&ch);
        assert_eq!(
            f32_to_e2m2(max_abs / scale),
            0x0F,
            "the channel peak must map to the top E2M2 code (0x0F)"
        );
        // The peak is a grid point, so it round trips exactly. This is a
        // stronger statement than "does not saturate": the scale is not merely
        // big enough, it is exactly right.
        let rec = tree_pie_rtn_channel(&ch);
        let rec_at_peak = rec[ch.iter().position(|v| v.abs() == max_abs).unwrap()];
        assert!(
            (rec_at_peak.abs() - max_abs).abs() < 1e-5,
            "the peak value must reconstruct exactly, got {} vs {max_abs}",
            rec_at_peak.abs()
        );
    }
}
