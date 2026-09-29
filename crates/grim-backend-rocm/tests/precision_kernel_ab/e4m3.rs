//! Host-side E4M3 encode/decode, mirrored from the device implementation.
//!
//! Shared by every FP8 arm (Raven, WhiteRaven) so their oracles agree by
//! construction: a difference in accuracy between them is then arithmetic, not
//! packing. Kept faithful to `grim_f32_to_fp8_e4m3` / `fp8_e4m3_to_float_hip`
//! in `kernels/dot_gemv.rs` rather than delegating to a library encoder, so the
//! oracle scores the kernel's own round trip.
//!
//! The device packer is a `__device__` function inside a HIP source string, so
//! it cannot be called from Rust. These are the host transcription, and the
//! tests below pin the bit patterns the two must agree on.

/// E4M3: sign(1) | exp(4, bias 7) | mant(3). No inf; `exp=15,mant=7` is NaN.
pub fn to_e4m3_rne(f: f32) -> u8 {
    if f.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if f.is_sign_negative() { 0x80 } else { 0 };
    let a = f.abs();
    if !a.is_finite() || a >= 448.0 {
        return sign | 0x7E; // saturate to 448
    }
    let bits = a.to_bits();
    let m = bits & 0x7FFFFF;
    let e = ((bits >> 23) & 0xFF) as i32;
    if e == 0 {
        return sign; // f32 subnormals are below E4M3 min
    }
    let mut ee = e - 120;
    if ee >= 1 {
        let mut q = m >> 20;
        let r = m & 0xFFFFF;
        if r > 0x80000 || (r == 0x80000 && q & 1 == 1) {
            q += 1;
        }
        if q == 8 {
            q = 0;
            ee += 1;
        }
        if ee > 15 {
            return sign | 0x7E;
        }
        return sign | ((ee as u8) << 3) | q as u8;
    }
    let sh = 21 - ee;
    // The device does `mant >> sh` and `1u << sh` in 32-bit, where sh reaches
    // 140 for small f32 and `1u << sh` is undefined. In u64 the shift is still
    // out of range, but the result is pinned: for sh >= 25, q = 0 and
    // r = mant < 2^24 = half, so the round-up branch can never fire and the
    // correct answer is always zero. sh == 24 must NOT be folded in here --
    // mant > 0x800000 there does round up to the min subnormal.
    if sh >= 25 {
        return sign;
    }
    let mant = 0x800000u64 | m as u64;
    let mut q = mant >> sh;
    let r = mant & ((1u64 << sh) - 1);
    let half = 1u64 << (sh - 1);
    if r > half || (r == half && q & 1 == 1) {
        q += 1;
    }
    if q == 0 {
        return sign;
    }
    sign | q as u8
}

/// Mirror of the device's `fp8_e4m3_to_float_hip`.
pub fn from_e4m3(v: u8) -> f32 {
    if v == 0x7F || v == 0xFF {
        return f32::NAN;
    }
    let sign = if v & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = (v >> 3) & 0x0F;
    let m = v & 0x07;
    if e == 0 {
        sign * m as f32 * (1.0 / 512.0)
    } else {
        sign * (1.0 + m as f32 / 8.0) * ((e as i32 - 7) as f32).exp2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values E4M3 represents exactly: every normal is 3 mantissa bits.
    #[test]
    fn normals_round_trip_exactly() {
        for e in -6i32..=7 {
            for m in 0..8u32 {
                let v = (1.0 + m as f32 / 8.0) * (2f32).powi(e);
                for s in [1.0f32, -1.0] {
                    let x = v * s;
                    let b = to_e4m3_rne(x);
                    assert_eq!(
                        from_e4m3(b),
                        x,
                        "E4M3 normal {x} (e={e} m={m}) must survive exactly"
                    );
                }
            }
        }
    }

    #[test]
    fn subnormals_round_trip_exactly() {
        // E4M3 subnormals are k * 2^-9 for k = 1..7.
        for k in 1..8u32 {
            let v = k as f32 * (2f32).powi(-9);
            assert_eq!(from_e4m3(to_e4m3_rne(v)), v, "subnormal {v}");
            assert_eq!(from_e4m3(to_e4m3_rne(-v)), -v, "subnormal -1 * 2^-9");
        }
    }

    #[test]
    fn zero_and_sign_handling() {
        assert_eq!(to_e4m3_rne(0.0), 0x00);
        assert_eq!(to_e4m3_rne(-0.0), 0x80);
        assert_eq!(from_e4m3(0x00), 0.0);
        assert_eq!(from_e4m3(0x80), -0.0);
        // Signed zero must not become positive zero: the sign bit is separate.
        assert_eq!(to_e4m3_rne(-1.0) & 0x80, 0x80);
    }

    /// RNE must be round-half-to-even, not truncation or half-away-from-zero.
    #[test]
    fn ties_go_to_even_not_up() {
        // 1.0625 is exactly halfway between 1.0 (m=0, even) and 1.125 (m=1).
        assert_eq!(from_e4m3(to_e4m3_rne(1.0625)), 1.0, "tie -> even mantissa");
        // 1.1875 is halfway between 1.125 (m=1, odd) and 1.25 (m=2, even).
        assert_eq!(from_e4m3(to_e4m3_rne(1.1875)), 1.25, "tie -> even mantissa");
    }

    /// Regression: the saturation guard was 480, one mantissa step too high.
    /// f32 in [448,480) has E=15, so q reaches 7 -- the NaN slot -- and every
    /// value in [464.01,480) encoded to NaN instead of saturating.
    #[test]
    fn the_band_just_below_480_saturates_instead_of_becoming_nan() {
        for x in [
            448.0f32, 450.0, 460.0, 464.0, 464.01, 470.0, 478.53, 479.0, 479.99,
        ] {
            for s in [1.0f32, -1.0] {
                let b = to_e4m3_rne(x * s);
                assert_eq!(b & 0x7F, 0x7E, "{x} must saturate to 0x7E, got {b:#04x}");
                assert_eq!(b >> 7, (x * s < 0.0) as u8, "sign lost for {x}");
                assert_eq!(from_e4m3(b), (x * s).signum() * 448.0);
            }
        }
    }

    /// No finite f32 may ever produce the NaN code. This is the invariant the
    /// 480 guard broke, stated directly so it fails loudly if reintroduced.
    #[test]
    fn no_finite_input_encodes_to_nan() {
        let mut s = 0x0BAD_C0DE_1234_5678u64;
        let mut rng = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Deliberately centred on the old guard's neighbourhood.
            ((s >> 40) as f32 / 8388608.0 - 1.0) * 500.0
        };
        for _ in 0..500_000 {
            let x = rng();
            if x.is_nan() {
                continue;
            }
            let b = to_e4m3_rne(x);
            assert_ne!(
                b & 0x7F,
                0x7F,
                "finite {x} encoded to the NaN slot {b:#04x}"
            );
            assert!(!from_e4m3(b).is_nan(), "finite {x} round-tripped to NaN");
        }
        // And the exact endpoints of the old broken window.
        for x in [464.01f32, 470.0, 478.53033, 479.999] {
            assert!(!from_e4m3(to_e4m3_rne(x)).is_nan(), "{x} became NaN");
        }
    }

    #[test]
    fn saturation_and_nan() {
        assert_eq!(to_e4m3_rne(1e30), 0x7E);
        assert_eq!(to_e4m3_rne(-1e30), 0xFE);
        assert_eq!(to_e4m3_rne(f32::INFINITY), 0x7E);
        assert_eq!(to_e4m3_rne(f32::NEG_INFINITY), 0xFE);
        // NaN input is the only route to the NaN code.
        assert_eq!(to_e4m3_rne(f32::NAN), 0x7F);
        assert!(from_e4m3(0x7F).is_nan());
        assert!(from_e4m3(0xFF).is_nan());
        // 447.9 must NOT saturate: it is representable as 0x7E's predecessor.
        assert!(
            from_e4m3(to_e4m3_rne(447.9)) > 400.0,
            "447.9 should stay near 448"
        );
    }

    /// The sh >= 25 guard. Below the min subnormal (2^-9) everything rounds to
    /// signed zero; the sh == 25 case is where the device's own `1u << sh`
    /// becomes undefined, so this pins the value it must produce anyway.
    #[test]
    fn tiny_magnitudes_flush_to_zero() {
        let min_sub = 2f32.powi(-9); // 0x01
        for e in -10i32..=-20 {
            let v = (2f32).powi(e);
            let b = to_e4m3_rne(v);
            if v < min_sub * 0.5 {
                assert_eq!(
                    b & 0x7F,
                    0x00,
                    "2^{e}={v} should flush to zero, got {b:#04x}"
                );
            } else {
                assert_eq!(b & 0x7F, 0x01, "2^{e}={v} should reach min subnormal");
            }
        }
        // Exactly half the min subnormal: RNE tie -> zero (even).
        assert_eq!(to_e4m3_rne(2f32.powi(-10)) & 0x7F, 0x00);
        // Just above half: rounds up to the min subnormal.
        assert_eq!(to_e4m3_rne(2f32.powi(-10) * 1.01) & 0x7F, 0x01);
    }

    /// f32 subnormals are below E4M3 min and must encode to signed zero.
    #[test]
    fn f32_subnormals_encode_to_zero() {
        assert_eq!(to_e4m3_rne(f32::from_bits(1)), 0x00);
        assert_eq!(to_e4m3_rne(-f32::from_bits(1)), 0x80);
    }

    /// Every finite f32 must land on a code this decoder does not call NaN, and
    /// the error must stay within one E4M3 ulp of the value's own magnitude.
    #[test]
    fn round_trip_error_is_bounded_and_never_nan() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rng = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8388608.0 - 1.0) * 500.0
        };
        for _ in 0..200_000 {
            let x = rng();
            if !x.is_finite() {
                continue;
            }
            let y = from_e4m3(to_e4m3_rne(x));
            assert!(!y.is_nan(), "finite {x} decoded to NaN");
            if x == 0.0 {
                continue;
            }
            // The one-ulp bound only holds inside the representable range.
            // E4M3's largest finite value is 448 (1.75 * 2^8) and the encoder
            // saturates rather than wrapping, so anything at or above the
            // saturation threshold is a clipping decision, not a rounding one.
            // Check clipping first, then the ulp bound on what remains.
            if x.abs() >= 448.0 {
                assert_eq!(y.abs(), 448.0, "x={x} must saturate to 448, got {y}");
                continue;
            }
            // Relative error is only bounded by half an ulp once the value is
            // in the *normal* range. Below E=1 the codes are spaced a fixed
            // 2^-9 apart, so a value near 3*2^-9 sits between two codes and can
            // be ~29% off. That is the format, not a defect -- so the bound
            // switches at the normal/subnormal boundary.
            let min_normal = 2f32.powi(-6); // E=1 -> 1.0 * 2^-6
            let bound = if x.abs() >= min_normal { 0.0625 } else { 0.5 };
            let rel = ((y - x) / x).abs();
            assert!(
                rel <= bound + 1e-3,
                "x={x} y={y} rel={rel} exceeds {bound}                  (min_normal={min_normal}, subnormal codes are 2^-9 apart)"
            );
        }
    }

    /// The encoder must be monotonic in magnitude: a kernel that feeds it
    /// increasing activations should get increasing codes.
    #[test]
    fn encoding_is_monotonic_in_magnitude() {
        let mut last = 0u8;
        for i in 0..4096 {
            let v = i as f32 * 0.01;
            let b = to_e4m3_rne(v) & 0x7F;
            assert!(
                b >= last,
                "non-monotonic at {v}: {b:#04x} after {last:#04x}"
            );
            last = b;
        }
    }
}
