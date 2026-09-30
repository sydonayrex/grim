//! TreePie: the FPE2M2 5-bit weight format (WS-A).
//!
//! # Format
//!
//! ```text
//! bit 4      : sign
//! bits 3..2  : exponent, bias 0
//! bits 1..0  : mantissa
//! ```
//!
//! No Inf, no NaN -- those encodings are reclaimed as ordinary numbers, which is
//! what lets a 2-bit exponent cover a useful range at all.
//!
//! ```text
//! e == 0 : value = (m / 2^2) * 2^(1-0) = m / 2      // subnormal
//! e >= 1 : value = (1 + m/4) * 2^e                  // normal
//! ```
//!
//! The `2^(1-b)` factor on the subnormal row is load-bearing. Dropping it gives
//! `0, .25, .5, .75, 2, 2.5, ...` -- a 2.67x hole between 0.75 and 2.0. With
//! it the grid is continuous, which is the only reason the format is worth
//! having:
//!
//! ```text
//! 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14
//! ```
//!
//! Range 28:1.
//!
//! # Why the decode is table-free
//!
//! FP16 has a 5-bit exponent with bias 15 and a 10-bit mantissa. The mapping
//! from E2M2 is therefore a pure shift-and-insert: add the bias to the exponent
//! field, and place the 2 mantissa bits at offset 8 (they are bits 8-9 of the
//! FP16 mantissa, the rest zero). No lookup table, no branch on the mantissa,
//! no integer multiply. This is the format's entire contribution -- see
//! `tests/tree_pie_codec_golden.rs` for the derivation and its oracle.
//!
//! The payoff is that TreePie's dequantized output *is* native FP16, so it feeds
//! `V_DOT2_F32_F16` / `V_DOT2C_F32_F16` directly rather than needing a new dot
//! instruction.

/// Bits per weight. 5 = 4-bit payload (8 values per i32) + 1 sign bit
/// (32 signs per i32), so 32 values occupy exactly 5 i32 = 5.0 bpw with zero
/// padding waste.
pub const TREE_PIE_BPW: f32 = 5.0;

/// Number of i32 words per 32 values: 4 payload + 1 sign plane.
pub const TREE_PIE_WORDS_PER_32: usize = 5;

/// Values per payload word: 8 nibbles.
const PER_PAYLOAD_WORD: usize = 8;

/// Payload words (the sign plane is the fifth).
const PAYLOAD_WORDS: usize = 4;

/// Pack 32 values into 5 i32 at exactly 5.0 bpw.
///
/// # Layout
///
/// ```text
/// words 0..3 : 4-bit payload (exp|mant), 8 nibbles per i32, low nibble first
/// word  4    : 1 sign bit per value, value i at bit i
/// ```
///
/// The planes are separated rather than interleaved because a 5-bit code does
/// not divide 32. Packing 6 codes per i32 the obvious way costs 5.33 bpw and
/// wastes 2 bits per word forever; isolating the sign plane recovers those bits
/// and is what makes the format exactly 5.0 bpw. It also gives the GPU decode
/// (A6) a nibble plane it can load directly and a sign mask it can apply with
/// one OR.
///
/// Total: never panics, and every output is a finite grid point. See
/// `f32_to_e2m2` for the saturation and NaN policy.
pub fn pack_tree_pie_32(values: &[f32; 32]) -> [i32; 5] {
    let mut payload = [0i32; PAYLOAD_WORDS];
    let mut signs = 0u32;

    for (i, &v) in values.iter().enumerate() {
        let code = f32_to_e2m2(v);
        payload[i / PER_PAYLOAD_WORD] |= ((code & 0x0F) as i32) << ((i % PER_PAYLOAD_WORD) * 4);
        if code & 0x10 != 0 {
            signs |= 1 << i;
        }
    }

    [
        payload[0],
        payload[1],
        payload[2],
        payload[3],
        signs as i32,
    ]
}

/// Inverse of [`pack_tree_pie_32`].
pub fn unpack_tree_pie_32(packed: &[i32; 5]) -> [f32; 32] {
    let signs = packed[4] as u32;
    let mut out = [0f32; 32];

    for (i, slot) in out.iter_mut().enumerate() {
        let nibble =
            ((packed[i / PER_PAYLOAD_WORD] as u32) >> ((i % PER_PAYLOAD_WORD) * 4)) & 0x0F;
        let sign = ((signs >> i) & 1) << 4;
        *slot = e2m2_to_f32(nibble as u8 | sign as u8);
    }

    out
}

/// Pack an arbitrary length slice, which must be a multiple of 32.
///
/// The multiple-of-32 requirement is structural rather than a convenience: the
/// 5.0 bpw claim only holds at the 32-value granularity, and silently rounding
/// up a ragged tail would overstate the density.
pub fn pack_tree_pie(values: &[f32]) -> Vec<i32> {
    assert!(
        values.len() % 32 == 0,
        "TreePie packs in blocks of 32, got {}",
        values.len()
    );
    let mut out = Vec::with_capacity(values.len() / 32 * TREE_PIE_WORDS_PER_32);
    for chunk in values.chunks(32) {
        let mut block = [0f32; 32];
        block.copy_from_slice(chunk);
        out.extend_from_slice(&pack_tree_pie_32(&block));
    }
    out
}

/// Inverse of [`pack_tree_pie`].
pub fn unpack_tree_pie(packed: &[i32]) -> Vec<f32> {
    let words = TREE_PIE_WORDS_PER_32;
    assert!(
        packed.len() % words == 0,
        "TreePie unpacks in blocks of {words} i32, got {}",
        packed.len()
    );
    let mut out = Vec::with_capacity(packed.len() / words * 32);
    for chunk in packed.chunks(words) {
        let mut block = [0i32; 5];
        block.copy_from_slice(chunk);
        out.extend_from_slice(&unpack_tree_pie_32(&block));
    }
    out
}

/// One TreePie code: `sign << 4 | exp << 2 | mant`.
pub type TreePieE2M2 = u8;

/// Decode a TreePie code to the **FP16 bit pattern** it denotes.
///
/// Table-free by construction: the only arithmetic is a bias add on the
/// exponent field and a shift of the mantissa into place. The sign is OR-ed in
/// last, which is why negative zero falls out for free rather than needing a
/// special case.
///
/// The `e == 0` row needs one conditional, and it is the *only* one: that row is
/// subnormal in E2M2 but is 0, 0.5, 1.0, 1.5 -- all of which are normal FP16
/// values, so it is a matter of picking the right exponent rather than
/// reconstructing a subnormal.
pub fn e2m2_to_fp16_bits(code: TreePieE2M2) -> u16 {
    let sign = ((code >> 4) & 1) as u16;
    let exp = (code >> 2) & 3;
    let mant = code & 3;

    // Exponent field in FP16, and mantissa already shifted to FP16 bit 8.
    //
    // e >= 1: bias 0 -> bias 15, so the field is just `exp + 15`.
    // e == 0: the subnormal row m/2 is 0, .5, 1, 1.5, whose FP16 exponent
    // fields are (only m=0 is special, being zero), 14, 15, 15. The two
    // distinct nonzero fields differ by 1, which is `mant >> 1` for mant in
    // 1..=3, with a +14 base.
    let (exp_field, mant_10) = if exp == 0 {
        if mant == 0 {
            (0u16, 0u16) // +-0
        } else {
            // m=1 -> .5 (field 14, mant 0), m=2 -> 1 (field 15, mant 0),
            // m=3 -> 1.5 (field 15, mant 512). The field is 14 + (m >> 1);
            // the mantissa LSB is set only at m=3, which is ((m+1)>>2) -- an
            // (m-1)&1 or m&1 term would put .5 at 1.25 and break the row.
            (
                14u16 + (mant >> 1) as u16,
                (((mant + 1) >> 2) as u16) << 9,
            )
        }
    } else {
        ((exp as u16) + 15, (mant as u16) << 8)
    };

    (sign << 15) | (exp_field << 10) | mant_10
}

/// Decode a TreePie code to its `f32` value.
///
/// Convenience over [`e2m2_to_fp16_bits`]; the value is exact in `f32` for every
/// code, since every grid point has at most 3 significant bits.
pub fn e2m2_to_f32(code: TreePieE2M2) -> f32 {
    f16_bits_to_f32(e2m2_to_fp16_bits(code))
}

/// Round `v` to the nearest TreePie code, ties to even, saturating.
///
/// Saturating rather than wrapping is a correctness requirement, not a
/// nicety: a wrapping encode turns a large outlier into a value of the wrong
/// magnitude and possibly the wrong sign, which is the single worst failure a
/// quantizer can have. It is also what keeps the encoder total, so it can never
/// panic on adversarial weights.
pub fn f32_to_e2m2(v: f32) -> TreePieE2M2 {
    if v.is_nan() {
        // No NaN encoding exists. Zero is the safe landing spot: NaN in a
        // weight is a caller bug, and encoding it as +-14 would amplify it.
        return 0;
    }
    let sign: u8 = if v.is_sign_negative() { 0x10 } else { 0 };
    let a = v.abs();

    if a >= 13.0 {
        // Midpoint between 12 and 14 is 13. Above that, 14 wins; and 14 is the
        // ceiling, so everything above it saturates too.
        return sign | 0x0F;
    }
    if a < 0.25 {
        // Midpoint between 0 and 0.5 is 0.25, which ties to even m=0 -> zero.
        return sign;
    }

    // Walk the 16 grid points and take the nearest, preferring the one with an
    // even mantissa on a tie. The grid is tiny and the comparison is exact in
    // f32, so a linear scan is both correct and not a hot path: the GPU does
    // the decode, not this.
    let mut best: u8 = 0;
    let mut best_err = f32::INFINITY;
    for code in 0..16u8 {
        let grid = e2m2_to_f32(code);
        let err = (grid - a).abs();
        // Strictly-less keeps the *earlier* code on a tie, and grid points
        // alternate odd/even mantissa as `code` increases, so "earlier" alone
        // does not give ties-to-even. Compare against the incumbent's parity
        // explicitly.
        if err < best_err {
            best = code;
            best_err = err;
        } else if err == best_err && (code & 1) == 0 && (best & 1) != 0 {
            best = code;
        }
    }
    sign | best
}

/// Expand `bits` (an FP16 bit pattern) to `f32`.
///
/// Local rather than pulled from a crate so the golden test's oracle and the
/// implementation under test share no helper that could mask a bug.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as f32;
    if exp == 0 {
        sign * mant * (1.0 / 1024.0) * (2f32).powi(-14)
    } else {
        sign * (1.0 + mant / 1024.0) * (2f32).powi(exp - 15)
    }
}

/// Dequantize TreePie from its **byte** storage form.
///
/// The codec is defined over `[i32; 5]` per 32 values, but a `Storage::FloatPack`
/// tensor hands back bytes, so this is the boundary the backends call. Rejects a
/// length that is not a whole number of groups and one whose groups are not
/// 5 i32 each, rather than truncating: a partial group would silently change the
/// element count, and TreePie's 5.0 bpw only holds at 32-value granularity.
///
/// Little-endian, matching the host: the ROCm kernel reads the same words as
/// 32-bit lanes, so host and device must agree on the byte order or the decode
/// diverges silently rather than loudly.
pub fn dequant_tree_pie_bytes(raw: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % 32 == 0,
        "TreePie needs n % 32 == 0 for the 5.0 bpw claim; got n = {n}"
    );
    let groups = n / 32;
    assert_eq!(
        raw.len(),
        groups * TREE_PIE_WORDS_PER_32 * 4,
        "TreePie storage must be {groups} groups of 5 i32 = {} bytes; got {}",
        groups * TREE_PIE_WORDS_PER_32 * 4,
        raw.len()
    );
    let mut out = Vec::with_capacity(n);
    let mut words = [0i32; TREE_PIE_WORDS_PER_32];
    for g in 0..groups {
        for (w, slot) in words.iter_mut().enumerate() {
            let o = (g * TREE_PIE_WORDS_PER_32 + w) * 4;
            *slot = i32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
        }
        out.extend_from_slice(&unpack_tree_pie_32(&words));
    }
    out
}

/// Pack `[f32]` into TreePie's byte storage form, the inverse of
/// [`dequant_tree_pie_bytes`].
pub fn pack_tree_pie_bytes(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0, "TreePie needs len % 32 == 0; got {}", values.len());
    let groups = values.len() / 32;
    let mut out = Vec::with_capacity(groups * TREE_PIE_WORDS_PER_32 * 4);
    for g in 0..groups {
        let mut block = [0f32; 32];
        block.copy_from_slice(&values[g * 32..g * 32 + 32]);
        for w in pack_tree_pie_32(&block) {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}
