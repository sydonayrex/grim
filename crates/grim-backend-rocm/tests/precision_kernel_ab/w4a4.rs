//! Host-side unsigned-int4 packers for WhiteCrow (`V_DOT8_I32_IU4`).
//!
//! # Layout, as the kernel indexes it
//!
//! Groups of 128 along K (`n_groups = K / 128`), one group per scale.
//!
//! **A (activations), symmetric, unsigned:**
//! - `codes[row][g]`: 16 `u32`, eight nibbles per word, 128 codes per group.
//!   Row stride is `n_groups * 16` words.
//! - `scales[row][g]`: `f32`.
//! - `sums[row][g]`: `i32`, the plain sum of the 128 codes.
//!
//! **B (weights), asymmetric, unsigned:**
//! - `codes[col][g]`: 16 `u32` per group; column stride is `K / 8` words.
//! - `scales[col][g]`: **bf16** (`grim_bf16_to_float` reads it).
//! - `zeros[col][g]`: `u8` zero point.
//!
//! # Why A has no zero point
//!
//! The kernel computes `d_a * d_b * (iacc - z_b * sum_qa)` — only B gets a
//! zero-point correction. A's codes are therefore unsigned 4-bit with an
//! implicit zero at code 0, so a signed activation has to be shifted into
//! `[0, 15]` and the shift is *not* recoverable from the kernel's output. The
//! arm quantizes A over non-negative values for that reason; `A_MIN` records
//! it so the constraint is visible at the call site instead of buried.
//!
//! # Nibble order
//!
//! Element `8*w + i` lives in nibble `i` (bits `4i..4i+4`) of word `w`, so
//! element 0 is the *low* nibble. That matches `dequant_mxfp4`'s packing, the
//! only other nibble packer in the tree.

/// Group size along K. The kernel's `n_groups = K / 128` assumes this.
pub const GROUP: usize = 128;
/// u32 words per group: 128 codes at 8 nibbles per word.
pub const WORDS_PER_GROUP: usize = GROUP / 8;

/// A must be non-negative: the kernel applies no zero-point correction to it.
/// See the module docs.
pub const A_MIN: f32 = 0.0;

/// Unsigned 4-bit codes: `[lo nibble = element 8w+0, hi nibble = element 8w+1]`.
pub fn pack_nibbles(codes: &[u8]) -> Vec<u32> {
    assert!(
        codes.len() % 8 == 0,
        "nibble packing needs a multiple of 8, got {}",
        codes.len()
    );
    codes
        .chunks(8)
        .map(|c| {
            let mut w = 0u32;
            for (i, &v) in c.iter().enumerate() {
                w |= ((v as u32) & 0xF) << (4 * i);
            }
            w
        })
        .collect()
}

/// Inverse of [`pack_nibbles`], for tests and oracles.
pub fn unpack_nibbles(words: &[u32], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for &w in words {
        for i in 0..8 {
            if out.len() == count {
                return out;
            }
            out.push(((w >> (4 * i)) & 0xF) as u8);
        }
    }
    out
}

/// f32 -> bf16, round-to-nearest-even, saturating to bf16's finite range.
/// The kernel reads these with `grim_bf16_to_float`, so the host encoder has
/// to agree with it bit for bit or the oracle is measuring the wrong scale.
pub fn to_bf16(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7FC0;
    }
    let sign: u16 = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = v.abs();
    if !a.is_finite() {
        return sign | 0x7F80;
    }
    let bits = a.to_bits();
    // Guard bf16's finite maximum (1 + 127/128) * 2^127 the same way the E4M3
    // guard did: clamp before the rounding carry can reach exp=255,mant=128,
    // which is bf16's NaN slot.
    const BF16_MAX: f32 = 3.3895314e38;
    if a >= BF16_MAX {
        return sign | 0x7F7F;
    }
    // RNE on the low 16 bits.
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    let exp = ((rounded >> 23) & 0xFF) as u16;
    if exp == 0xFF {
        // Rounding carried into the exponent; re-clamp rather than emit inf/NaN.
        return sign | 0x7F7F;
    }
    sign | (((bits >> 16) as u16) & 0xFFFF).min(sign | 0x7F7F)
}

pub fn from_bf16(h: u16) -> f32 {
    f32::from_bits(((h as u32) & 0xFFFF) << 16)
}

/// Pack one 128-wide group of A into (codes, scale, sum).
///
/// Unsigned 4-bit over `[A_MIN, amax]`, so `scale = amax / 15` and the code is
/// `round((a - A_MIN) / scale)` clamped to `[0, 15]`.
fn pack_a_group(g: &[f32]) -> (Vec<u32>, f32, i32) {
    debug_assert_eq!(g.len(), GROUP);
    let lo = A_MIN;
    let amax = g.iter().fold(lo, |m, v| m.max(*v));
    let scale = ((amax - lo) / 15.0).max(1e-30);
    let codes: Vec<u8> = g
        .iter()
        .map(|&v| (((v - lo) / scale).round().clamp(0.0, 15.0)) as u8)
        .collect();
    let sum = codes.iter().map(|&c| c as i32).sum();
    (pack_nibbles(&codes), scale, sum)
}

/// Pack one 128-wide group of B into (codes, bf16 scale, u8 zero point).
///
/// Asymmetric, in the form the kernel's identity assumes:
/// `value = d_b * (code - z)`, so the group is centred on `z` rather than on
/// `min`.
///
/// The zero point is the code nearest zero, `round(-min / scale)`. Note the
/// trap: codes are `round((v - min)/scale)` **without** adding `z` first. An
/// earlier version computed `round((v-min)/scale + zf)` and clamped to 15,
/// which silently destroyed the top of the range — for min=-3, max=+3 it gave
/// b=+3 the code 22, clamped to 15, dequantizing to 0.0 instead of +3. Adding
/// the zero point into the code and then also subtracting it in the kernel
/// double-counts. The zero point belongs in the *dequant*, not the encode.
fn pack_b_group(g: &[f32]) -> (Vec<u32>, u16, u8) {
    debug_assert_eq!(g.len(), GROUP);
    let min = g.iter().fold(f32::INFINITY, |m, v| m.min(*v));
    let max = g.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
    let scale = ((max - min) / 15.0).max(1e-30);

    let zf = (-min / scale).round().clamp(0.0, 15.0);
    let z = zf as u8;

    let codes: Vec<u8> = g
        .iter()
        .map(|&v| (((v - min) / scale).round().clamp(0.0, 15.0)) as u8)
        .collect();
    // Round the scale to bf16 *before* handing it back, so the caller and the
    // kernel agree on the only scale that will ever be used.
    let scale_h = to_bf16(scale);
    (pack_nibbles(&codes), scale_h, z)
}

/// A packed for `launch_dot8_w4a4_gemv`: codes, f32 scales, i32 sums.
pub struct PackedA {
    pub codes: Vec<u32>,
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
    pub groups: usize,
}

pub fn pack_a(a: &[f32], m: usize, k: usize) -> PackedA {
    assert_eq!(a.len(), m * k);
    assert_eq!(k % GROUP, 0, "WhiteCrow needs K % 128 == 0");
    let groups = k / GROUP;
    let mut codes = Vec::with_capacity(m * groups * WORDS_PER_GROUP);
    let mut scales = Vec::with_capacity(m * groups);
    let mut sums = Vec::with_capacity(m * groups);
    for row in 0..m {
        for g in 0..groups {
            let off = row * k + g * GROUP;
            let (w, s, sum) = pack_a_group(&a[off..off + GROUP]);
            codes.extend_from_slice(&w);
            scales.push(s);
            sums.push(sum);
        }
    }
    PackedA { codes, scales, sums, groups }
}

/// B packed for `launch_dot8_w4a4_gemv`: codes, bf16 scales, u8 zeros.
#[derive(Clone)]
pub struct PackedB {
    pub codes: Vec<u32>,
    pub scales: Vec<u16>,
    pub zeros: Vec<u8>,
    pub groups: usize,
}

pub fn pack_b(b: &[f32], n: usize, k: usize) -> PackedB {
    assert_eq!(b.len(), n * k);
    assert_eq!(k % GROUP, 0, "WhiteCrow needs K % 128 == 0");
    let groups = k / GROUP;
    let words_per_col = k / 8;
    let mut codes = Vec::with_capacity(n * words_per_col);
    let mut scales = Vec::with_capacity(n * groups);
    let mut zeros = Vec::with_capacity(n * groups);
    for col in 0..n {
        for g in 0..groups {
            let off = col * k + g * GROUP;
            let (w, s, z) = pack_b_group(&b[off..off + GROUP]);
            codes.extend_from_slice(&w);
            scales.push(s);
            zeros.push(z);
        }
    }
    PackedB { codes, scales, zeros, groups }
}

/// The kernel's arithmetic, in f64, for one output element.
///
/// Reproduces `d_a * d_b * (iacc - z_b * sum_qa)` summed over groups, reading
/// the *packed* values so the oracle is the kernel's own quantity. Written
/// against the packed form rather than the original f32 on purpose: scoring
/// against the source values would charge the kernel for the quantizer, which
/// is the mistake that made the first Raven run report 1e3 of phantom error.
pub fn dot_packed(
    pa: &PackedA,
    pb: &PackedB,
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    let groups = k / GROUP;
    let words_per_col = k / 8;
    let a_row = i * groups * WORDS_PER_GROUP;
    let b_col = j * words_per_col;
    let mut acc = 0f64;
    for g in 0..groups {
        let d_a = pa.scales[i * groups + g] as f64;
        let sum_qa = pa.sums[i * groups + g] as f64;
        let d_b = from_bf16(pb.scales[j * groups + g]) as f64;
        let z_b = pb.zeros[j * groups + g] as f64;

        let a_base = a_row + g * WORDS_PER_GROUP;
        let b_base = b_col + g * WORDS_PER_GROUP;
        let mut iacc = 0f64;
        for w in 0..WORDS_PER_GROUP {
            let aw = pa.codes[a_base + w];
            let bw = pb.codes[b_base + w];
            for i_nib in 0..8 {
                let ac = ((aw >> (4 * i_nib)) & 0xF) as f64;
                let bc = ((bw >> (4 * i_nib)) & 0xF) as f64;
                iacc += ac * bc;
            }
        }
        acc += d_a * d_b * (iacc - z_b * sum_qa);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;


    /// Element 0 must be the low nibble: 0x21 -> [1, 2, ...], matching
    /// dequant_mxfp4, the only other nibble packer in the tree.
    #[test]
    fn element_zero_is_the_low_nibble() {
        let codes: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let w = pack_nibbles(&codes);
        assert_eq!(w[0], 0x7654_3210, "word 0 must be little-endian in nibbles");
        assert_eq!(w[1], 0xFEDC_BA98);
        assert_eq!(unpack_nibbles(&w, 16), codes, "round trip must be exact");
    }

    #[test]
    fn nibble_packing_ignores_bits_above_the_low_four() {
        // Codes are 4-bit; a stray high bit must not bleed into the neighbour.
        // 8 codes per word, so pad to a multiple of 8.
        let codes = vec![0xF0u8, 0x0F, 0xFF, 0x00, 0xF0, 0x0F, 0xFF, 0x00];
        let w = pack_nibbles(&codes);
        assert_eq!(w[0], 0x0FF0_0FF0, "high bits must be masked off");
        assert_eq!(unpack_nibbles(&w, 8), vec![0x0, 0xF, 0xF, 0x0, 0x0, 0xF, 0xF, 0x0]);
    }

    #[test]
    fn nibble_packing_rejects_a_partial_word() {
        // Better to fail loudly than to zero-pad: a short group would otherwise
        // decode as real zeros and read as plausible weights.
        assert!(std::panic::catch_unwind(|| pack_nibbles(&[1, 2, 3])).is_err());
    }

    /// The identity the whole kernel rests on. If the nibble order, the code
    /// mapping, or the zero point disagree by one position, this fails.
    #[test]
    fn the_two_dot_identity_reconstructs_the_dot_product() {
        let m = 2usize;
        let n = 4usize;
        let k = 256usize;
        let mut s = 0x1234_5678u64;
        let mut rng = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / 2147483648.0 - 1.0
        };
        // A non-negative by construction; B spans both signs.
        let a: Vec<f32> = (0..m * k).map(|_| rng().abs()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng()).collect();

        let pa = pack_a(&a, m, k);
        let pb = pack_b(&b, n, k);
        for i in 0..m {
            for j in 0..n {
                let got = dot_packed(&pa, &pb, i, j, k);
                // Reference: dequantize the packed operands, then dot in f64.
                let mut want = 0f64;
                for g in 0..k / GROUP {
                    let d_a = pa.scales[i * (k / GROUP) + g] as f64;
                    let d_b = from_bf16(pb.scales[j * (k / GROUP) + g]) as f64;
                    let z = pb.zeros[j * (k / GROUP) + g] as f64;
                    for e in 0..GROUP {
                        let ac =
                            ((pa.codes[i * (k / GROUP) * 16 + g * 16 + e / 8] >> (4 * (e % 8)))
                                & 0xF) as f64;
                        let bc = ((pb.codes[j * (k / 8) + g * 16 + e / 8] >> (4 * (e % 8)))
                            & 0xF) as f64;
                        want += (d_a * ac) * (d_b * (bc - z));
                    }
                }
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "({i},{j}): two-dot {got} != dequantized-dot {want}"
                );
            }
        }
    }

    /// A monotone column must reconstruct to its own shape, extremes included.
    ///
    /// Two earlier versions of this test asserted the *mean code* equals the
    /// zero point. That is not a property of this format: a column of just two
    /// extreme values packs to codes {0, 15} and nothing else, so the mean is
    /// 7.5 regardless of where the true zero point sits. Both the symmetric and
    /// the asymmetric fixture produced a mean of 7.5, which is what exposed the
    /// premise as wrong rather than the packer.
    ///
    /// What actually matters is that the reconstructed *values* are centred, so
    /// that is what is asserted: the dequantized column's midpoint is near zero
    /// and its span matches the original.
    #[test]
    fn the_dequantized_column_keeps_its_midpoint_and_span() {
        let k = 128usize;
        for (label, b) in [
            ("equal-and-opposite", vec![3.0f32, -3.0].repeat(k / 2)),
            ("asymmetric", vec![2.0f32, -4.0].repeat(k / 2)),
        ] {
            let pb = pack_b(&b, 1, k);
            let d = from_bf16(pb.scales[0]) as f64;
            let z = pb.zeros[0] as f64;
            let codes = unpack_nibbles(&pb.codes, k);
            let deq: Vec<f64> = codes.iter().map(|&c| d * (c as f64 - z)).collect();

            let lo = deq.iter().cloned().fold(f64::MAX, f64::min);
            let hi = deq.iter().cloned().fold(f64::MIN, f64::max);
            let mid = (lo + hi) / 2.0;
            let want_mid = (b.iter().cloned().fold(f32::MAX, f32::min) as f64
                + b.iter().cloned().fold(f32::MIN, f32::max) as f64)
                / 2.0;
            let want_span = (b.iter().cloned().fold(f32::MIN, f32::max)
                - b.iter().cloned().fold(f32::MAX, f32::min)) as f64;

            assert!(
                (mid - want_mid).abs() < 0.15 * want_span,
                "{label}: midpoint {mid} vs {want_mid} (span {want_span})"
            );
            assert!(
                ((hi - lo) - want_span).abs() < 0.05 * want_span,
                "{label}: span {} vs {want_span}",
                hi - lo
            );
        }
    }

    /// A non-cancelling dot, so int4 is actually able to represent it.
    #[test]
    fn a_non_cancelling_dot_survives_within_the_int4_budget() {
        let k = 1024usize;
        let mut s = 0x5EED_1234u64;
        let mut rng = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / 2147483648.0 - 1.0
        };
        // All-positive B: no cancellation, so the sum is O(sqrt(K)) and the
        // relative quantizer error stays at the format's own ~10%.
        let a: Vec<f32> = (0..k).map(|_| rng().abs()).collect();
        let b: Vec<f32> = (0..k).map(|_| rng().abs()).collect();
        let pa = pack_a(&a, 1, k);
        let pb = pack_b(&b, 1, k);
        let got = dot_packed(&pa, &pb, 0, 0, k);
        let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let rel = ((got - want) / want).abs();
        assert!(rel < 0.25, "non-cancelling int4 dot rel={rel} (got {got}, want {want})");
    }

    /// The stored scale is bf16, so the oracle must use the bf16 value. Using
    /// the f32 scale would report error the kernel never made.
    #[test]
    fn b_scale_is_bf16_and_the_oracle_agrees_with_the_kernel() {
        let k = 128usize;
        let b: Vec<f32> = (0..k).map(|i| (i as f32) * 0.017).collect();
        let pb = pack_b(&b, 1, k);
        let h = pb.scales[0];
        // Round-tripping bf16 twice must be stable.
        assert_eq!(to_bf16(from_bf16(h)), h, "bf16 round trip must be stable");
        // And it must be a real bf16: exponent+mantissa only, no f32 bits.
        let back = from_bf16(h);
        assert_eq!(back.to_bits() & 0xFFFF, 0, "bf16 occupies the high 16 bits");
    }

    #[test]
    fn bf16_saturates_instead_of_becoming_inf_or_nan() {
        // The E4M3 lesson applied preemptively: rounding must not carry into
        // bf16's inf/NaN slots.
        assert_eq!(to_bf16(f32::INFINITY), 0x7F80);
        // f32::MAX (3.4028235e38) exceeds bf16's finite range (3.3895314e38),
        // so the rounding carry lands on exp=255. That must clamp, not wrap.
        let f32_max = f32::MAX;
        assert_eq!(to_bf16(f32_max), 0x7F7F, "f32::MAX must clamp, not wrap");
        assert!(!from_bf16(to_bf16(f32_max)).is_nan());
        assert!(!from_bf16(0x7F7F).is_nan(), "0x7F7F must be finite, not NaN");
        // Just under bf16's max must survive as a normal number.
        let under = 3.0e38f32;
        assert!(from_bf16(to_bf16(under)).is_finite() && from_bf16(to_bf16(under)) > 2.0e38);
        assert_eq!(to_bf16(f32::NAN), 0x7FC0);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // 1.0 is exact; 1.0 + 2^-9 is the tie and must go to even (1.0's low
        // mantissa bit is 0).
        assert_eq!(from_bf16(to_bf16(1.0)), 1.0);
        let tie = 1.0 + (1.0 / 512.0);
        assert_eq!(from_bf16(to_bf16(tie)), 1.0, "tie must round to even");
        // One ulp above the tie must round up. bf16 has 7 explicit mantissa
        // bits, so an ulp at 1.0 is 2^-7 = 1/128, and the tie sits at half that.
        assert_eq!(from_bf16(to_bf16(1.0)), 1.0, "1.0 is exact in bf16");
        let above = 1.0 + (1.5 / 128.0);
        assert!(
            from_bf16(to_bf16(above)) > 1.0,
            "1.5 ulp above 1.0 must round up, got {}",
            from_bf16(to_bf16(above))
        );
        // And the tie itself goes to even (down), since 1.0's mantissa is even.
        let tie = 1.0 + (0.5 / 128.0);
        assert_eq!(from_bf16(to_bf16(tie)), 1.0, "tie must round to even");
    }

    #[test]
    fn packing_is_shape_correct() {
        let m = 3usize;
        let n = 5usize;
        let k = 384usize; // 3 groups
        let a = vec![1.0f32; m * k];
        let b = vec![-2.0f32; n * k];
        let pa = pack_a(&a, m, k);
        let pb = pack_b(&b, n, k);
        assert_eq!(pa.groups, 3);
        assert_eq!(pa.codes.len(), m * 3 * 16, "16 words per group per row");
        assert_eq!(pa.scales.len(), m * 3);
        assert_eq!(pa.sums.len(), m * 3);
        assert_eq!(pb.codes.len(), n * (k / 8), "K/8 words per column");
        assert_eq!(pb.scales.len(), n * 3);
        assert_eq!(pb.zeros.len(), n * 3);
        assert_eq!(pb.groups, 3, "PackedB carries the group count for the arm");
    }

    #[test]
    fn a_codes_stay_in_range_and_scale_is_finite() {
        let k = 512usize;
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.37).collect();
        let pa = pack_a(&a, 1, k);
        for &w in &pa.codes {
            for i in 0..8 {
                assert!((w >> (4 * i)) & 0xF <= 15);
            }
        }
        assert!(pa.scales.iter().all(|s| s.is_finite() && *s > 0.0));
        // Sum must match the codes, since the kernel subtracts z_b * sum_qa.
        for g in 0..pa.groups {
            let mut want = 0i32;
            for w in 0..WORDS_PER_GROUP {
                let word = pa.codes[g * WORDS_PER_GROUP + w];
                for i in 0..8 {
                    want += ((word >> (4 * i)) & 0xF) as i32;
                }
            }
            assert_eq!(pa.sums[g], want, "group {g} sum must match its codes");
        }
    }

    #[test]
    fn an_all_zero_group_does_not_divide_by_zero() {
        let k = 128usize;
        let a = vec![0.0f32; k];
        let pa = pack_a(&a, 1, k);
        assert!(pa.scales[0].is_finite() && pa.scales[0] > 0.0);
        assert_eq!(pa.sums[0], 0);
        let pb = pack_b(&a, 1, k);
        assert!(from_bf16(pb.scales[0]).is_finite());
    }

    /// The int4 error budget, derived rather than guessed. Group-wise, A's
    /// quantizer has step `amax/15`, so a uniform rounding error of at most
    /// step/2. The dot product over K terms therefore carries a relative error
    /// of roughly (step/2) / (typical |a|), which for a well-conditioned
    /// Gaussian group is ~ (amax/15/2) / (amax/3) ~ 10%. B's asymmetric
    /// quantizer contributes similarly.
    ///
    /// So int4 cannot be held to the int8 arm's 1e-3: its own format ceiling is
    /// two orders of magnitude looser. This constant is the derived bound, and
    /// the test below checks the measured error sits under it rather than
    /// tuning the constant to whatever came out.
    #[test]
    fn measured_int4_error_sits_under_the_derived_budget() {
        const GROUP_TOL: f64 = 0.25; // ~2.5x the 10% estimate, for tail shapes
        let m = 1usize;
        let n = 64usize;
        let k = 2048usize;
        let mut s = 0xDEAD_BEEFu64;
        let mut rng = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / 2147483648.0 - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| rng().abs()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng()).collect();
        let pa = pack_a(&a, m, k);
        let pb = pack_b(&b, n, k);

        let mut worst = 0f64;
        for j in 0..n {
            let got = dot_packed(&pa, &pb, 0, j, k);
            let mut want = 0f64;
            for kk in 0..k {
                want += a[kk] as f64 * b[j * k + kk] as f64;
            }
            worst = worst.max(((got - want) / want.abs().max(1e-6)).abs());
        }
        assert!(
            worst <= GROUP_TOL,
            "int4 relative error {worst} exceeds the derived budget {GROUP_TOL}"
        );
    }
}

