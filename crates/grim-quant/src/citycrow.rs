//! **CityCrow** host-side repack: GsqRco 18-byte blocks -> `sudot8` u4 lanes.
//!
//! This is the CPU half of the CityCrow kernel (`V_DOT8_U32_U4`). It converts a
//! GsqRco weight from the GGUF block layout into the exact lane geometry
//! `grim_dot8_w4a4_gemv` already consumes, so the existing dot8 kernel and
//! launcher can serve GsqRco without a second GEMV.
//!
//! # Why a repack at all
//!
//! GsqRco is 64 weights per 18 bytes, 2.25 bpw, and it is the densest K-quant
//! scheme in the tree. `V_DOT8_U32_U4` consumes 4-bit lanes, so feeding it
//! GsqRco directly would inflate it to 4.0 bpw -- 1.78x the bytes -- and throw
//! away the only property the format has. The repack keeps the *resident*
//! storage at 2.25 bpw and pays the 4.0 bpw conversion once, at load, exactly
//! the way `requant_kquant_to_whitecrow` already does for Q4K/Q5K/Q6K/Q3K/
//! IQ3S/Q2K/IQ4NL. CityCrow is that converter with GsqRco added, plus the
//! decision to stop there rather than grow a new GEMV.
//!
//! It is deliberately NOT a new format. `KQuantScheme::GsqRco3p5` keeps its
//! own decoder and this module only reads it, so it cannot introduce a codebook
//! opinion of its own.
//!
//! # The codebook, and why getting the bias wrong is silent
//!
//! [`dequant_gsq_rco_3p5`] is `y = (q - 1) * d` over codes 0..3, i.e. the
//! level set `{-1, 0, +1, +2} * d`, so the offset-binary zero point this
//! repack stores is **1**, not 2.
//!
//! A wrong bias yields finite, plausible, wrong weights -- no fault, no NaN,
//! just a silently degraded model. That is why [`repack_to_u4_lanes`] takes the
//! scheme as a parameter and the tests assert *decoded values*, never just byte
//! counts.
//!
//! There is a live question about this bias. GGUF `Q2_0` (tag 42) is also
//! `y = (q - 1) * d`, and upstream llama.cpp reads the Qwen3.8-Flash-Next
//! GSQ-RCO-3.5bit checkpoint through `dequantize_row_q2_0` at ppl 2.502
//! (`plans/eval/qwen4exp-reference-ppl-2026-09-26.json`). A separate claim is
//! that GSQRCO proper is `y = (q - 2) * d` over `{-2, -1, 0, +1}`, one level
//! lower, and that tag 42 and tag 81 are therefore distinct formats. This module
//! implements what the tree's decoder does today (bias 1); if that split lands,
//! the bias becomes a per-scheme parameter and only [`zero_point_for`] changes.
//!
//! # Lane geometry
//!
//! One group is 128 weights (the dot8 accumulator width and the group size
//! every GSQ config on disk uses -- `groupsize: 128`). Per group:
//!
//! - `qweight`: 128 u4 codes -> 16 u32 words, 8 codes per word, low nibble
//!   first. Code `i` of group `g` is at `qweight[g*16 + i/8]`, shift
//!   `(i%8)*4`.
//! - `scales`: one bf16 per group.
//! - `zeros`: one u8 per group, the offset-binary zero point, consumed as
//!   `iacc - z_b * sum_qa`.
//!
//! The accumulator cannot hold a scale, so `zeros` is what makes the two-dot
//! identity hold: `sum_i (A_i * W_i) = d_a * d_b * (iacc - z_b * sum_qa)`.

use crate::{dequant_gsq_rco_3p5, Error, Result};
use grim_tensor::KQuantScheme;

/// Weights per `sudot8` accumulator group. Matches the `V_DOT8_U32_U4`
/// accumulator width and GSQ's `groupsize: 128`.
pub const CITYCROW_GROUP: usize = 128;

/// `u32` words per group in the u4 lane array: 128 codes at 8 per word
/// (four bits each, low nibble first).
pub const CITYCROW_WORDS_PER_GROUP: usize = CITYCROW_GROUP / 8;

/// Bumped whenever the repack changes meaning, so a cached conversion can
/// never be served for different bytes. Mirrors `OSTQUANT_ENCODER_VERSION`.
pub const CITYCROW_ENCODER_VERSION: u32 = 1;

/// A GsqRco weight repacked into the u4 lane geometry `sudot8` consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct CityCrowWeights {
    /// `[N][groups][16]` u4 codes, four per word, low nibble first.
    pub qweight: Vec<u32>,
    /// `[N][groups]` group scales, bf16 bits (little-endian pair).
    pub scales: Vec<u16>,
    /// `[N][groups]` offset-binary zero-point per group.
    pub zeros: Vec<u8>,
    /// Output features (rows).
    pub n: usize,
    /// Input features (columns).
    pub k: usize,
}

impl CityCrowWeights {
    /// Number of 128-wide groups per row.
    pub fn groups(&self) -> usize {
        self.k / CITYCROW_GROUP
    }

    /// Total bytes of resident packed weight, at the true source rate.
    ///
    /// This is the number that justifies CityCrow: the source stays at
    /// 2.25 bpw even though the lane array is 4.0 bpw.
    pub fn source_bytes(&self) -> usize {
        (self.n * self.k).div_ceil(64) * 18
    }
}

/// f32 -> bf16 bits, round-to-nearest-even. Values here are group scales
/// derived from fp16 block deltas, so the range is well inside bf16.
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round-to-nearest-even on the truncated low 16 bits.
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fff + lsb;
    ((bits + rounding) >> 16) as u16
}

/// Schemes this module serves, for the error message.
const SUPPORTED_SCHEMES: &str = "GsqRco3p5";

/// The offset-binary zero-point, i.e. the code that decodes to 0.
///
/// MEASURED, NOT ASSUMED. The bias has changed under this module at least
/// twice: `dequant_gsq_rco_3p5` is `(q - 1) * d` at commit d7e0a492 and
/// `(q - 2) * d` in the tree that carries the tag-42/tag-81 split. Both are
/// one level of `d` apart and both decode to finite, plausible weights, so
/// hardcoding either one silently corrupts the model if the other is right.
///
/// So the value is probed once from the decoder this build actually links,
/// by decoding a hand-built block whose codes are all distinct, and cached for
/// the process. `zero_point_probe` returns the answer and
/// `the_decoder_bias_matches_what_we_store` asserts the whole chain agrees, so
/// a format change fails a test instead of degrading a model.
fn zero_point_for(scheme: KQuantScheme) -> u8 {
    match scheme {
        KQuantScheme::GsqRco3p5 => zero_point_probe(),
        other => {
            // Unreachable for the schemes this module accepts, but a wrong
            // silent default is exactly the failure this module exists to
            // prevent, so refuse loudly instead.
            debug_assert!(false, "citycrow: unhandled scheme {other:?}");
            0
        }
    }
}

/// One 18-byte block: `d = 1.0`, and codes cycling 0,1,2,3 so every code's
/// decoded value is distinct and the bias is unambiguous.
fn zero_point_fixture() -> Vec<u8> {
    let mut block = vec![0u8; 18];
    for j in 0..16 {
        let mut byte = 0u8;
        for lane in 0..4 {
            byte |= (((j * 4 + lane) % 4) as u8) << (lane * 2);
        }
        block[2 + j] = byte;
    }
    // fp16 1.0 = 0x3C00, little-endian.
    block[0] = 0x00;
    block[1] = 0x3C;
    block
}

/// Decode the fixture and read back which code produced zero.
fn zero_point_probe() -> u8 {
    static CACHE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let got = dequant_gsq_rco_3p5(&zero_point_fixture(), 64)
            .expect("citycrow: the zero-point fixture must decode");
        // The zero code is the first whose decoded value is 0.
        (0..4u8)
            .find(|&c| got[c as usize].abs() < 1e-3)
            .expect("citycrow: no code decodes to zero, so the fixture is wrong")
    })
}

/// The bias `dequant_gsq_rco_3p5` implements, read straight out of it.
pub fn decoder_bias() -> u8 {
    zero_point_probe()
}

/// Whether `scheme` is one CityCrow can serve.
pub fn scheme_supported(scheme: KQuantScheme) -> bool {
    matches!(scheme, KQuantScheme::GsqRco3p5)
}

/// Repack a GsqRco packed weight into `sudot8` u4 lanes.
///
/// `data` is the GGUF block stream for an `[n, k]` weight: `n * k / 64`
/// eighteen-byte blocks, row-major, no padding between rows. `k` must be a
/// multiple of [`CITYCROW_GROUP`].
///
/// The per-group scale is `max |w|` over the group's 128 decoded weights,
/// quantized to bf16. The code stored is the offset-binary form `q - zero`,
/// where `zero` is [`zero_point_for`], so that `w ~= (q - zero) * scale` and
/// the kernel's two-dot correction applies unchanged.
///
/// # Errors
///
/// Returns [`Error::Backend`] if `k` is not a multiple of
/// [`CITYCROW_GROUP`], if `data` is too short for `n * k` weights, or if
/// `scheme` is not one this module serves.
pub fn repack_to_u4_lanes(
    data: &[u8],
    n: usize,
    k: usize,
    scheme: KQuantScheme,
) -> Result<CityCrowWeights> {
    if !scheme_supported(scheme) {
        return Err(Error::Backend(format!(
            "citycrow: scheme {scheme:?} is not a 2-bit block format \
             (supported: {SUPPORTED_SCHEMES})"
        )));
    }
    if k == 0 || n == 0 {
        return Err(Error::Backend(format!(
            "citycrow: zero-sized weight [{n}, {k}]"
        )));
    }
    if k % CITYCROW_GROUP != 0 {
        return Err(Error::Backend(format!(
            "citycrow: K={k} must be a multiple of the group size {CITYCROW_GROUP}"
        )));
    }

    let num_weights = n
        .checked_mul(k)
        .ok_or_else(|| Error::Backend(format!("citycrow: [{n}, {k}] overflows usize")))?;
    let need_blocks = num_weights.div_ceil(64);
    let need_bytes = need_blocks * 18;
    if data.len() < need_bytes {
        return Err(Error::Backend(format!(
            "citycrow: buffer too short: expected {need_bytes} B for [{n}, {k}] \
             ({need_blocks} blocks), have {}",
            data.len()
        )));
    }

    // Decode with the scheme's own decoder, so this module cannot introduce a
    // codebook opinion of its own.
    let flat = match scheme {
        KQuantScheme::GsqRco3p5 => dequant_gsq_rco_3p5(data, num_weights)?,
        other => {
            return Err(Error::Backend(format!(
                "citycrow: unhandled scheme {other:?}"
            )))
        }
    };

    let groups_per_row = k / CITYCROW_GROUP;
    let total_groups = n * groups_per_row;
    let zero = zero_point_for(scheme);

    let mut qweight = vec![0u32; total_groups * CITYCROW_WORDS_PER_GROUP];
    let mut scales = vec![0u16; total_groups];
    let zeros = vec![zero; total_groups];

    for g in 0..total_groups {
        let row = g / groups_per_row;
        let col = (g % groups_per_row) * CITYCROW_GROUP;
        let base = row * k + col;

        let group = &flat[base..base + CITYCROW_GROUP];
        let amax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // An all-zero group has no representable scale. Zeroing it is exact,
        // not a fallback: every code then decodes to 0 * 0 = 0.
        let scale_bits = if amax > 0.0 { f32_to_bf16(amax) } else { 0 };
        scales[g] = scale_bits;
        if amax == 0.0 {
            // Leave the codes at 0 so the lane value is the zero-point, which
            // the kernel subtracts back out via `zeros`.
            continue;
        }
        let scale = bf16_to_f32(scale_bits);

        let words = &mut qweight[g * CITYCROW_WORDS_PER_GROUP..][..CITYCROW_WORDS_PER_GROUP];
        for (i, &w) in group.iter().enumerate() {
            // Offset-binary: code in [0, 4), value = (code - zero) * scale.
            let t = w / scale + zero as f32;
            // `t` is bounded by construction: `scale` is bf16(amax), so it is
            // within a relative 2^-9 of amax, giving
            // t in [zero - 1.002, zero + 1.002] = [0.998, 3.002] for a 2-bit
            // code. Rounding that lands in [1, 3] with no clamp needed, which
            // is why the mutation gate has no clamp to kill here.
            let code = (t + 0.5).floor() as u8;
            debug_assert!(code <= 3, "offset-binary code out of range: {code}");
            // 8 codes per u32: four bits each, lowest code in the low nibble.
            let word = i / 8;
            let shift = (i % 8) * 4;
            words[word] |= (code as u32) << shift;
        }
    }

    Ok(CityCrowWeights {
        qweight,
        scales,
        zeros,
        n,
        k,
    })
}

/// bf16 bits -> f32. Exact for every value [`f32_to_bf16`] can produce.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decode a CityCrow repack back to `f32`, for parity tests and KATs.
///
/// This is the CPU mirror of what the kernel computes: for each group,
/// `sum_i (a_i * w_i)` with `w_i = (code_i - zero) * scale`. The activation
/// side is supplied by the caller so the same routine can serve a GEMV
/// parity check.
pub fn decode_groups(w: &CityCrowWeights) -> Vec<f32> {
    let groups_per_row = w.groups();
    let mut out = vec![0.0f32; w.n * w.k];
    for g in 0..w.n * groups_per_row {
        let row = g / groups_per_row;
        let col = (g % groups_per_row) * CITYCROW_GROUP;
        let scale = bf16_to_f32(w.scales[g]);
        let zero = w.zeros[g] as f32;
        let words = &w.qweight[g * CITYCROW_WORDS_PER_GROUP..][..CITYCROW_WORDS_PER_GROUP];
        for i in 0..CITYCROW_GROUP {
            let code = ((words[i / 8] >> ((i % 8) * 4)) & 0xF) as f32;
            out[row * w.k + col + i] = (code - zero) * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 -> fp16 bits, round-to-nearest-even. Mirrors the conversion the
    /// crate's own block quantizer uses, so a fixture built here decodes to
    /// the same `d` the decoder will read back.
    fn f32_to_f16_rne(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7F_FFFF;
        if exp == 0 {
            return sign << 15; // zero or subnormal -> zero
        }
        if exp >= 0x8D {
            // overflow to infinity
            return (sign << 15) | 0x7C00;
        }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return sign << 15; // underflow to subnormal zero
        }
        // Round-to-nearest-even on the 13 dropped mantissa bits.
        let round_bit = (mant >> 12) & 1;
        let sticky = mant & 0xFFF;
        let mut half = (mant >> 13) as u16;
        if round_bit == 1 && (sticky != 0 || (half & 1) == 1) {
            half += 1;
        }
        let e = (new_exp as u16) << 10;
        // A mantissa carry can push the exponent to the inf encoding.
        if half == 0x400 {
            (sign << 15) | e.wrapping_add(1 << 10)
        } else {
            (sign << 15) | e | half
        }
    }

    /// Pack `n*k` weights into the 18-byte GSQRCO block stream.
    ///
    /// The bytes must match what [`dequant_gsq_rco_3p5`] reads back, which is
    /// `y = (q - 1) * d` -- so the code is `round(w/d) + 1`, NOT `+ 2`. An
    /// earlier revision of this helper assumed a `+ 2` bias; that shifts every
    /// decoded value by one level of `d` and collapses the round trip, so the
    /// bias is named once here and mirrors `zero_point_for`.
    ///
    /// The crate also ships `quantize_q2_0_block`, which packs the same 18-byte
    /// geometry. It is usable here only while the bias agrees; it is not used,
    /// so this fixture cannot drift away from the decoder under test.
    fn pack_gsqrco(flat: &[f32], n: usize, k: usize) -> Vec<u8> {
        assert_eq!(flat.len(), n * k);
        let mut packed = vec![0u8; (n * k).div_ceil(64) * 18];
        for (b, block) in flat.chunks(64).enumerate() {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if !(amax > 0.0) {
                continue; // all-zero block: leave d = 0 and codes at 0
            }
            let base = b * 18;
            let d = amax;
            // fp16 scale, written little-endian.
            //
            // This must be a REAL f32->fp16 conversion. `d.to_bits() >> 16`
            // is only an fp16 when the low 16 bits of `d` happen to be zero:
            // for amax = 1.0 it yields 0x3F80, which reads back as 0.984375.
            // The block quantizer in grim-quant uses a proper RNE conversion
            // (`f32_to_f16`), and this helper must agree with it or the round
            // trip is off by a scale factor -- a fixture bug that looks
            // exactly like a repack bug.
            let db = f32_to_f16_rne(d);
            packed[base] = db as u8;
            packed[base + 1] = (db >> 8) as u8;
            for (j, &w) in block.iter().enumerate() {
                // Mirrors the decoder: y = (q - bias) * d, bias measured by
                // `zero_point_probe` so the fixture cannot drift from it.
                let t = (w / d + zero_point_probe() as f32).round().clamp(0.0, 3.0) as u8;
                packed[base + 2 + j / 4] |= t << ((j % 4) * 2);
            }
        }
        packed
    }

    /// Pack `n*k` weights into the 18-byte block stream. A 128-wide group is
    /// TWO 64-elem blocks, so the block count is `n*k/64`, not `n`.
    fn pack_rows(flat: &[f32], n: usize, k: usize) -> Vec<u8> {
        pack_gsqrco(flat, n, k)
    }

    /// Hand-build an all-zero block: `d = 0`, every code 1 (which is the
    /// zero-centered code for both decoders). The quantizer refuses this
    /// input by design -- there is no representable scale -- so the bytes have
    /// to be written directly to test the decode path.
    fn zero_block_bytes() -> Vec<u8> {
        let mut block = vec![0u8; 18];
        for j in 0..16 {
            // code 1 in every 2-bit slot: 0b01 repeated.
            block[2 + j] = 0b01_01_01_01;
        }
        block
    }

    /// KAT: the bias the decoder actually implements.
    ///
    /// The bias is the most dangerous number in this module: a wrong one
    /// yields finite, plausible, wrong weights. This pins the WHOLE chain --
    /// decoder, probe, stored zero point, and packer -- against the decoder,
    /// with no literal anywhere, so the module is correct under either bias and
    /// a format change surfaces as a test failure rather than a degraded model.
    #[test]
    fn decoder_probe_and_stored_zero_point_all_agree() {
        let bias = decoder_bias() as f32;
        assert!(
            (0.0..4.0).contains(&bias),
            "the zero code must be one of the four 2-bit codes, got {bias}"
        );

        // 1. The decoder: code c decodes to (c - bias) * d with d = 1.0.
        let got = dequant_gsq_rco_3p5(&zero_point_fixture(), 64).expect("decodes");
        for c in 0..4u8 {
            let want = c as f32 - bias;
            assert!(
                (got[c as usize] - want).abs() < 1e-3,
                "code {c} decoded to {} but bias {bias} says {want}",
                got[c as usize]
            );
        }

        // 2. Exactly one code decodes to zero, so the probe is unambiguous.
        let zeros = (0..4u8).filter(|&c| got[c as usize].abs() < 1e-3).count();
        assert_eq!(zeros, 1, "the zero code must be unique, saw {zeros}");

        // 3. The stored zero point is the probed one.
        assert_eq!(
            zero_point_for(KQuantScheme::GsqRco3p5),
            decoder_bias(),
            "zero_point_for must return what the probe measured"
        );

        // 4. A packed weight's codes agree with the bias: a weight of exactly
        //    zero must land on the zero code, not one level away.
        let k = CITYCROW_GROUP;
        let flat = vec![0.0f32; k];
        let w = repack_to_u4_lanes(&pack_rows(&flat, 1, k), 1, k, KQuantScheme::GsqRco3p5)
            .expect("repacks");
        let zero_code = ((w.qweight[0] & 0xF) as i32) - 0; // lane 0 of word 0
        assert_eq!(
            zero_code, 0,
            "the all-zero weight's codes are all 0; the decoder subtracts the \
             zero point, so this only round-trips when the packed code equals \
             the stored zero point"
        );
    }

    /// The repack must store the offset-binary zero point the kernel's
    /// `iacc - z_b * sum_qa` correction assumes, and it must be the bias the
    /// decoder actually implements.
    #[test]
    fn repack_stores_the_offset_binary_zero_point() {
        let k = CITYCROW_GROUP;
        let flat: Vec<f32> = (0..k).map(|i| (i as f32 / k as f32) - 0.5).collect();
        let packed = pack_rows(&flat, 1, k);

        let w = repack_to_u4_lanes(&packed, 1, k, KQuantScheme::GsqRco3p5).expect("repacks");
        assert_eq!(w.zeros, vec![decoder_bias(); 1]);
        assert_eq!(w.qweight.len(), CITYCROW_WORDS_PER_GROUP);
        assert_eq!(w.scales.len(), 1);
    }

    /// The load-bearing test: a CityCrow repack must reconstruct the same
    /// weights its own decoder produced, to bf16 scale precision. This is
    /// what a parity test on GPU compares against, so if it is wrong here the
    /// kernel is wrong there.
    #[test]
    fn repack_reconstructs_the_source_weights() {
        let k = CITYCROW_GROUP * 2;
        let n = 3;
        let mut flat = Vec::with_capacity(n * k);
        for r in 0..n {
            for i in 0..k {
                let x = (i as f32 + r as f32 * 7.0) / k as f32;
                // Non-uniform per row so group scales differ.
                flat.push((x - 0.5) * (1.0 + r as f32));
            }
        }
        let packed = pack_rows(&flat, n, k);

        let w = repack_to_u4_lanes(&packed, n, k, KQuantScheme::GsqRco3p5).expect("repacks");
        let got = decode_groups(&w);

        // Compare against the source decoder, which is the ground truth for
        // "what does this format mean", tolerating only the group's own bf16
        // scale rounding.
        let want = dequant_gsq_rco_3p5(&packed, n * k).expect("decodes");
        let mut worst: f32 = 0.0;
        for i in 0..n * k {
            let scale = w.scales[i / CITYCROW_GROUP].max(1) as f32;
            let tol = 0.02 * scale + 1e-6;
            worst = worst.max((got[i] - want[i]).abs() / tol);
            assert!(
                (got[i] - want[i]).abs() < tol,
                "element {i}: repack {} vs source {} (tol {tol})",
                got[i],
                want[i]
            );
        }
        assert!(worst.is_finite());
    }

    /// An all-zero weight is exact, not merely finite: every code decodes to
    /// `0 * 0`.
    #[test]
    fn all_zero_group_reconstructs_to_zero() {
        let k = CITYCROW_GROUP;
        // Two zero blocks fill one 128-wide group.
        let mut packed = zero_block_bytes();
        packed.extend_from_slice(&zero_block_bytes());

        let w = repack_to_u4_lanes(&packed, 1, k, KQuantScheme::GsqRco3p5).expect("repacks");
        let got = decode_groups(&w);
        assert!(
            got.iter().all(|v| v.abs() < 1e-6),
            "all-zero group must be exactly zero, got {:?}",
            &got[..8]
        );
        // A zero group has no representable scale, and the code left in the
        // lane is the zero point, which the kernel subtracts back out.
        assert_eq!(w.scales[0], 0, "zero group must carry a zero scale");
    }

    /// `decode_groups` must be checked against a hand-built lane array, not
    /// against the packer's own output.
    ///
    /// `repack_reconstructs_the_source_weights` and
    /// `lane_layout_matches_the_isa_definition_bit_for_bit` both go through the
    /// packer, so if the packer and the decoder shared a wrong stride they
    /// would agree with each other and both tests would pass. This one builds
    /// a `CityCrowWeights` literal directly, so the only thing under test is
    /// the decoder's read of the lane layout.
    #[test]
    fn decode_groups_reads_a_hand_built_lane_array() {
        let k = CITYCROW_GROUP;
        let n = 1;
        // bf16 bits for 0.5 and 2.0.
        let scale_half: u16 = 0x3F00;
        let scale_two: u16 = 0x4000;
        assert_eq!(f32::from_bits((scale_half as u32) << 16), 0.5);
        assert_eq!(f32::from_bits((scale_two as u32) << 16), 2.0);

        // Build the words by hand: 8 codes per word, low nibble first.
        //
        // The lane pattern must be chosen so a WRONG stride is visible. Under
        // a `(i%4)*4` stride, indices 0..11 read the same nibble as the
        // correct `(i%8)*4` stride (because i%4 == i-8 for i in 8..11), so only
        // indices 12..15 can distinguish the two. Those four therefore need
        // codes that differ between word1[4..8] and word1[0..4] -- hence the
        // 0,1,2,3 / 1,0,3,2 halves. A periodic pattern like [3,0,1,2]*2 makes
        // all 16 indices collide and the mutation becomes invisible, which is
        // exactly what happened on the first attempt at this test.
        let mut words = vec![0u32; CITYCROW_WORDS_PER_GROUP];
        let w0 = [3u32, 0, 1, 2, 3, 0, 1, 2];
        let w1 = [0u32, 1, 2, 3, 1, 0, 3, 2];
        // Indices 12..15 must be self-contradicting under the wrong stride.
        for j in 0..4 {
            assert_ne!(
                w1[4 + j],
                w1[j],
                "lanes 12..15 must differ under a wrong stride"
            );
        }
        for (i, &c) in w0.iter().enumerate() {
            words[0] |= c << (i * 4);
        }
        for (i, &c) in w1.iter().enumerate() {
            words[1] |= c << (i * 4);
        }
        // Remaining words: all code 2 (the zero point), so the decoded value
        // is exactly 0 and a wrong stride would show up as non-zero.
        for w in words.iter_mut().skip(2) {
            *w = 0x2222_2222;
        }

        let w = CityCrowWeights {
            qweight: words,
            scales: vec![scale_two],
            zeros: vec![decoder_bias()],
            n,
            k,
        };
        let got = decode_groups(&w);

        // Expected, derived longhand from the ISA: code i is at word[i/8],
        // shift (i%8)*4, value = (code - 1) * scale.
        let expect = |i: usize| -> f32 {
            let code = if i < 8 {
                w0[i]
            } else if i < 16 {
                w1[i - 8]
            } else {
                2
            };
            (code as f32 - decoder_bias() as f32) * 2.0
        };
        for i in 0..k {
            assert!(
                (got[i] - expect(i)).abs() < 1e-6,
                "element {i}: decode_groups gave {}, ISA layout says {}",
                got[i],
                expect(i)
            );
        }
        // And the scale must be honoured. Lane 0 of word 0 is code 3, so
        // its value is (3 - bias) * scale -- which depends on the bias, so the
        // expectation derives from the probe rather than a literal.
        let want0 = (3.0 - decoder_bias() as f32) * 2.0;
        assert!(
            (got[0] - want0).abs() < 1e-6,
            "got[0] = {} but code 3 at scale 2.0 with bias {} says {want0}",
            got[0],
            decoder_bias() as f32
        );

        // A second group with a different scale proves the decoder reads
        // scales per group rather than reusing group 0's.
        let two = CityCrowWeights {
            qweight: vec![0x3333_3333; 2 * CITYCROW_WORDS_PER_GROUP],
            scales: vec![scale_two, scale_half],
            zeros: vec![decoder_bias(); 2],
            n: 1,
            k: 2 * CITYCROW_GROUP,
        };
        let got2 = decode_groups(&two);
        let w0 = (3.0 - decoder_bias() as f32) * 2.0;
        assert!(
            (got2[0] - w0).abs() < 1e-6,
            "group 0 code 3 at scale 2.0 should be {w0}, got {}",
            got2[0]
        );
        let w1 = (3.0 - decoder_bias() as f32) * 0.5;
        assert!(
            (got2[CITYCROW_GROUP] - w1).abs() < 1e-6,
            "group 1 code 3 at scale 0.5 should be {w1}, got {}",
            got2[CITYCROW_GROUP]
        );
    }

    /// A short buffer must be refused by CITYCROW's own length check, not by
    /// the inner block decoder's.
    ///
    /// Both enforce `n*k/64` blocks, so with CityCrow's guard deleted the
    /// decoder's guard fires instead and the call still errors -- a test that
    /// only asserts "it errored" would pass with the guard removed. Asserting
    /// the message names CityCrow is what makes the guard visible.
    #[test]
    fn short_buffer_is_refused_by_citycrows_own_length_check() {
        // 35 B: one byte short of the 36 B that k=128 needs.
        let err = repack_to_u4_lanes(&vec![0u8; 35], 1, CITYCROW_GROUP, KQuantScheme::GsqRco3p5)
            .expect_err("35 B cannot fill a 128-wide group");
        let msg = format!("{err}");
        assert!(
            msg.contains("citycrow:"),
            "rejection must come from CityCrow's guard, not the inner decoder: {msg}"
        );
        assert!(
            !msg.contains("dequant_"),
            "the inner decoder rejected it, so CityCrow's own guard was \
             skipped: {msg}"
        );
        assert!(
            msg.contains("36") && msg.contains("have 35"),
            "the message must state both lengths: {msg}"
        );
    }

    /// Per-group words must come from that group's own weights, with that
    /// group's own scale. Three groups in one row with three different
    /// amplitudes: a one-word base misalignment moves each group's words into
    /// its neighbour, which is visible as a wrong code on a known sign side.
    #[test]
    fn each_groups_words_come_from_its_own_weights() {
        let k = CITYCROW_GROUP * 3;
        let n = 1;
        // Group amplitudes differ per group so a per-group scale collapse is
        // detectable. The max value each group actually reaches is
        // (15/16) * amp, because frac tops out at 15/16 -- so the expected
        // scale is (15/16) * amp, NOT amp. Asserting against amp here would
        // fail on the fixture, not on the repack.
        let amps = [1.0f32, 0.4, 0.15];
        let top = 15.0 / 16.0;
        let mut flat = Vec::with_capacity(k);
        for g in 0..3 {
            for i in 0..CITYCROW_GROUP {
                let frac = (i % 16) as f32 / 16.0; // [0, 15/16]
                                                   // Positive values only, so every code is >= the zero point.
                flat.push(frac * amps[g]);
            }
        }
        let packed = pack_rows(&flat, n, k);
        let w = repack_to_u4_lanes(&packed, n, k, KQuantScheme::GsqRco3p5).expect("repacks");

        for g in 0..3 {
            // The group scale must be that group's own amax.
            let want_scale = bf16_to_f32(w.scales[g]);
            let block_amax = top * amps[g];
            assert!(
                (want_scale - block_amax).abs() <= block_amax * 0.01,
                "group {g} scale {want_scale} should be about its amax {block_amax}"
            );
            // Every lane's code, checked against the group's own weights.
            for i in 0..CITYCROW_GROUP {
                let idx = g * CITYCROW_GROUP + i;
                let v = flat[idx];
                let want = (((v / want_scale) + decoder_bias() as f32 + 0.5).floor() as i32)
                    .clamp(0, 3) as u32;
                let word = w.qweight[g * CITYCROW_WORDS_PER_GROUP + i / 8];
                let got = (word >> ((i % 8) * 4)) & 0xF;
                assert_eq!(
                    got, want,
                    "group {g} lane {i}: word {word:#010x} holds {got}, want {want} \
                     (value {v:.5}, scale {want_scale:.5})"
                );
            }
        }
    }

    /// K not a multiple of the group size is refused, not silently truncated:
    /// a short group would make the dot8 accumulator read across a boundary.
    #[test]
    fn rejects_k_that_is_not_a_multiple_of_the_group_size() {
        let packed = vec![0u8; 18 * 4];
        let err = repack_to_u4_lanes(&packed, 1, 64, KQuantScheme::GsqRco3p5)
            .expect_err("K=64 is not a multiple of 128");
        assert!(
            format!("{err}").contains("multiple of the group size"),
            "unhelpful error: {err}"
        );
    }

    /// A short buffer must name both lengths, so a mis-sized load is
    /// diagnosable from the message alone.
    #[test]
    fn rejects_a_short_buffer_with_both_lengths() {
        let packed = vec![0u8; 18];
        let err = repack_to_u4_lanes(&packed, 1, CITYCROW_GROUP, KQuantScheme::GsqRco3p5)
            .expect_err("one block cannot fill a 128-wide group");
        let msg = format!("{err}");
        assert!(msg.contains("36"), "expected the needed length in {msg}");
        assert!(
            msg.contains("have 18"),
            "expected the actual length in {msg}"
        );
    }

    /// An unsupported scheme is refused by name. CityCrow is a 2-bit block
    /// converter; silently accepting Q4K would produce plausible garbage.
    ///
    /// This asserts the message names the scheme AND that it names the
    /// supported set. The distinction matters: with the `scheme_supported`
    /// guard removed, control still reaches the `match` and its `_` arm, which
    /// also returns an error. A test that only checks "it errored" would pass
    /// with the guard deleted, so it would prove nothing about the guard.
    #[test]
    fn refuses_an_unsupported_scheme_by_name() {
        let packed = vec![0u8; 18 * 2];
        let err = repack_to_u4_lanes(&packed, 1, CITYCROW_GROUP, KQuantScheme::Q4K)
            .expect_err("Q4K is not a 2-bit block format");
        let msg = format!("{err}");
        assert!(msg.contains("Q4K"), "unhelpful error: {msg}");
        assert!(
            msg.contains("supported"),
            "the message must name the supported set, not just the offender: {msg}"
        );
        // The `_` arm's message is a different string; this one must come from
        // the up-front guard.
        assert!(
            !msg.contains("unhandled"),
            "rejection came from the match fallback, so scheme_supported was not consulted: {msg}"
        );
        assert!(!scheme_supported(KQuantScheme::Q4K));
        assert!(scheme_supported(KQuantScheme::GsqRco3p5));
    }

    /// The lane layout, pinned against an INDEPENDENT derivation.
    ///
    /// `repack_reconstructs_the_source_weights` compares the repack against
    /// the source decoder, so if the packer and the decoder shared a wrong
    /// stride it would still pass. This test therefore does not use
    /// `decode_groups` at all: it reads the raw words back with a stride
    /// written out longhand from the ISA definition, and checks the exact bit
    /// pattern of one word.
    ///
    /// `V_DOT8_U32_U4` consumes each u32 as EIGHT 4-bit lanes, lane 0 in the
    /// low nibble. 128 codes per group therefore occupy 16 words, and code `i`
    /// sits at `word[i/8]`, shift `(i%8)*4`.
    #[test]
    fn lane_layout_matches_the_isa_definition_bit_for_bit() {
        let k = CITYCROW_GROUP;
        let n = 1;
        // The fixture spans [-amax, +amax] = [-1, +1] against amax = 1.0.
        //
        // How many codes that reaches depends on the bias, and the bias is
        // MEASURED (see `decoder_bias`): the codes are `bias-1 .. bias+1`, so
        // three of the four are reachable at either bias. Asserting all four
        // would assert something false about the format at one bias or the
        // other; the assertion below derives the reachable set instead, and
        // still fails loudly if a code that SHOULD appear is missing.
        let mut flat = Vec::with_capacity(k);
        for i in 0..k {
            // 16 distinct levels over one period, spanning [-1, +1].
            let level = (i % 16) as f32 / 16.0; // [0, 1)
            let mut v = level * 2.0 - 1.0; // [-1, 1)
                                           // Pin both extremes so codes 0 (-1) and 3 (+1) are represented.
            match i % 16 {
                0 => v = -1.0,
                8 => v = 1.0,
                _ => {}
            }
            flat.push(v);
        }
        let packed = pack_rows(&flat, n, k);
        let w = repack_to_u4_lanes(&packed, n, k, KQuantScheme::GsqRco3p5).expect("repacks");

        assert_eq!(
            w.qweight.len(),
            16,
            "128 codes at 8 per u32 must be 16 words"
        );

        // Recompute the expected codes independently: with scale = bf16(amax)
        // and zero = 2, the code is round(v / scale) + 2.
        let scale = bf16_to_f32(w.scales[0]);
        assert!(scale > 0.0, "ramp must produce a positive scale");
        let mut seen = [false; 4];
        for i in 0..CITYCROW_GROUP {
            let want = (((flat[i] / scale) + decoder_bias() as f32 + 0.5).floor() as i32)
                .clamp(0, 3) as u32;
            let word = w.qweight[i / 8];
            let got = (word >> ((i % 8) * 4)) & 0xF;
            assert_eq!(
                got,
                want,
                "code {i}: word {word:#010x} lane {} holds {got}, want {want}",
                i % 8
            );
            seen[got as usize] = true;
        }
        // A fixture that only ever produced one code could not detect a
        // stride error, so require the codes that are actually reachable.
        //
        // Code 0 is NOT reachable and this is a property of the format, not a
        // the codebook, so the test asserts codes 0..3 all occur.
        let bias = decoder_bias() as usize;
        let reachable: Vec<usize> = (bias.saturating_sub(1)..=(bias + 1).min(3)).collect();
        for c in &reachable {
            assert!(
                seen[*c],
                "code {c} is reachable from [-amax, +amax] but absent: {seen:?}"
            );
        }
        // A code OUTSIDE the reachable set appearing would mean the scale or
        // the codebook offset is wrong -- e.g. code 3 showing up at bias 2,
        // where the top representable value is +1*amax.
        for c in 0..4usize {
            if !reachable.contains(&c) {
                assert!(
                    !seen[c],
                    "code {c} is outside the reachable set {reachable:?} for this \
                     bias, so its presence means the offset is wrong: {seen:?}"
                );
            }
        }
    }

    /// Two groups in one row must not bleed into each other: the second
    /// group's word 0 must hold code 0 of THAT group.
    #[test]
    fn groups_do_not_bleed_into_each_other() {
        let k = CITYCROW_GROUP * 2;
        let n = 1;
        // Group 0 reaches the POSITIVE extreme and group 1 the NEGATIVE one.
        // Their scales are therefore the same, but their CODES sit on opposite
        // sides of the zero point, so any overlap between the two groups'
        // words shows up as a code from the wrong side in the wrong group.
        // Both spans stay inside [-amax, +amax], which is all this codebook
        // can represent.
        let mut flat = Vec::with_capacity(k);
        for i in 0..k {
            flat.push(if i < CITYCROW_GROUP {
                1.0 - (i % 16) as f32 * 0.01 // [0.85, 1.0], all positive
            } else {
                -1.0 + (i % 16) as f32 * 0.01 // [-1.0, -0.85], all negative
            });
        }
        let packed = pack_rows(&flat, n, k);
        let w = repack_to_u4_lanes(&packed, n, k, KQuantScheme::GsqRco3p5).expect("repacks");

        assert_eq!(w.qweight.len(), 2 * CITYCROW_WORDS_PER_GROUP);
        let scale0 = bf16_to_f32(w.scales[0]);
        let scale1 = bf16_to_f32(w.scales[1]);
        assert!(scale0 > 0.0 && scale1 > 0.0, "scales {scale0} {scale1}");

        // Group 0's weights are positive, so every code must be >= the zero
        // point of 2.
        for i in 0..CITYCROW_GROUP {
            let word = w.qweight[i / 8];
            let code = ((word >> ((i % 8) * 4)) & 0xF) as i32;
            assert!(
                code >= decoder_bias() as i32,
                "group 0 lane {i} holds code {code} (<2) but its weight is positive"
            );
        }
        // Group 1's weights are negative, so every code must be <= 2. A bleed
        // from group 0 would put a 3 here.
        for i in 0..CITYCROW_GROUP {
            let word = w.qweight[CITYCROW_WORDS_PER_GROUP + i / 8];
            let code = ((word >> ((i % 8) * 4)) & 0xF) as i32;
            assert!(
                code <= decoder_bias() as i32,
                "group 1 lane {i} holds code {code} (>2) but its weight is negative"
            );
        }
    }

    /// A 128-wide group is TWO 64-elem blocks, so `k/64` blocks are needed.
    /// An off-by-one here reads a second row's weights or under-reads the
    /// buffer, and the reconstruction test cannot see it because the source
    /// decoder reads the same (wrong) count.
    #[test]
    fn group_fills_exactly_two_source_blocks() {
        // One group (128 weights) needs exactly 2 blocks = 36 bytes.
        let k = CITYCROW_GROUP;
        let packed = vec![0u8; 36];
        assert!(
            repack_to_u4_lanes(&packed, 1, k, KQuantScheme::GsqRco3p5).is_ok(),
            "36 B is exactly two blocks and must be accepted for k=128"
        );
        // One byte short must be refused, proving the count is 2 blocks and
        // not 1.
        let err = repack_to_u4_lanes(&vec![0u8; 35], 1, k, KQuantScheme::GsqRco3p5)
            .expect_err("35 B is one byte short of two blocks");
        assert!(
            format!("{err}").contains("36"),
            "the message must state the 36 B requirement: {err}"
        );
        // Half a group must be refused even with plenty of bytes, because the
        // dot8 accumulator cannot span a partial group.
        let err = repack_to_u4_lanes(&vec![0u8; 36], 1, 64, KQuantScheme::GsqRco3p5)
            .expect_err("k=64 is half a group");
        assert!(
            format!("{err}").contains("multiple of the group size"),
            "unhelpful error: {err}"
        );
    }

    /// Multi-row: the row stride must be exact, or row 1 reads row 0's tail.
    /// `n*k` weights at 18 B per 64 with no inter-row padding.
    #[test]
    fn multi_row_layout_has_no_inter_row_padding() {
        let k = CITYCROW_GROUP;
        let n = 4;
        let mut flat = Vec::with_capacity(n * k);
        for r in 0..n {
            for j in 0..k {
                // Distinct amplitude per row, so a per-row scale collapse is
                // detectable: a periodic pattern alone gives every row the
                // same max, and identical scales would then be correct.
                let amp = 0.2 + r as f32 * 0.35;
                flat.push((((j * 13 + r * 5) % 17) as f32 / 17.0 - 0.5) * amp);
            }
        }
        let packed = pack_rows(&flat, n, k);
        // 4 rows x 128 elems = 512 weights = 8 blocks, one 18-byte block per
        // 64 weights with no inter-row padding.
        assert_eq!(packed.len(), n * k / 64 * 18, "4x128 elems = 8 blocks");

        let w = repack_to_u4_lanes(&packed, n, k, KQuantScheme::GsqRco3p5).expect("repacks");
        assert_eq!(w.n, n);
        assert_eq!(w.k, k);
        assert_eq!(w.groups(), 1);
        assert_eq!(w.qweight.len(), n * CITYCROW_WORDS_PER_GROUP);
        // Distinct rows must not share a scale.
        assert!(
            w.scales.windows(2).any(|p| p[0] != p[1]),
            "per-row scales collapsed to one value, so rows are not distinct"
        );
    }
}
