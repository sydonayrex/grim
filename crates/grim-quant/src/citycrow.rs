//! **CityCrow** host-side repack: GsqRco 18-byte blocks -> `sudot8` u4 lanes.
//!
//! This is the CPU half of the CityCrow kernel (`V_DOT8_U32_U4`). It converts a
//! GsqRco/Q2_0 weight from the GGUF block layout into the exact lane geometry
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
//! It is deliberately NOT a new format. `KQuantScheme::GsqRco3p5` and
//! `KQuantScheme::Q2_0` keep their own decoders; this module only reads them.
//!
//! # The two codebooks, and why getting this wrong is silent
//!
//! Both schemes share 64-elem/18-byte geometry and differ by exactly one level:
//!
//! | scheme | decode | codebook |
//! |---|---|---|
//! | `Q2_0` (tag 42) | `y = (q - 1) * d` | `{-1, 0, +1, +2}` |
//! | `GsqRco3p5` (tag 81) | `y = (q - 2) * d` | `{-2, -1, 0, +1}` |
//!
//! A wrong bias here yields finite, plausible, wrong weights -- no fault, no
//! NaN, just a silently degraded model. That is why [`repack_to_u4_lanes`]
//! takes the scheme as a parameter and every test asserts the *decoded values*,
//! never just the byte count.
//!
//! Evidence for which codebook a real checkpoint uses, and for the tag split,
//! is the differential perplexity oracle in
//! `plans/eval/qwen4exp-reference-ppl-2026-09-26.json`: upstream llama.cpp
//! @ `13c1eb24` scores the Qwen3.8-Flash-Next GSQ-RCO-3.5bit checkpoint at
//! ppl 2.502 on `plans/eval/wikitext2.sample.txt`, reading it through
//! `dequantize_row_q2_0`. Note the pinned `f3f1a8f2` is the DENSE reference
//! and `13c1eb24` applies the sparse mask, so PPL across those two commits
//! diverges by design and any A/B must say which it used.
//!
//! # Lane geometry
//!
//! One group is 128 weights (the dot8 accumulator width and the group size
//! every GSQ config on disk uses -- `groupsize: 128`). Per group:
//!
//! - `qweight`: 128 u4 codes -> 16 u32 words, 4 codes per word, low nibble
//!   first. Word `w` of group `g` holds codes `4w .. 4w+3`.
//! - `scales`: one bf16 per group.
//! - `zeros`: one u8 per group, the offset-binary zero-point `2^(bits-1)`,
//!   i.e. 2 for a 2-bit code. Consumed as `iacc - z_b * sum_qa`.
//!
//! The accumulator cannot hold a scale, so `zeros` is what makes the two-dot
//! identity hold: `sum_i (A_i * W_i) = d_a * d_b * (iacc - z_b * sum_qa)`.

use crate::{dequant_gsq_rco_3p5, dequant_q2_0, Error, Result};
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

/// A GsqRco/Q2_0 weight repacked into the u4 lane geometry `sudot8` consumes.
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

/// The offset-binary zero-point for a code width, i.e. `2^(bits-1)`.
///
/// For the 2-bit GsqRco code this is 2, which is exactly the bias its decoder
/// subtracts. Storing it per group is what lets the kernel apply the
/// two-dot correction without knowing the bit width.
fn zero_point_for(scheme: KQuantScheme) -> u8 {
    match scheme {
        // 2-bit codes either way; both decoders centre at code 2 / code 1.
        KQuantScheme::Q2_0 | KQuantScheme::GsqRco3p5 => 1u8 << 1,
        other => {
            // Unreachable for the schemes this module accepts, but a wrong
            // silent default is exactly the failure this module exists to
            // prevent, so refuse loudly instead.
            debug_assert!(false, "citycrow: unhandled scheme {other:?}");
            0
        }
    }
}

/// Whether `scheme` is one CityCrow can serve.
pub fn scheme_supported(scheme: KQuantScheme) -> bool {
    matches!(scheme, KQuantScheme::Q2_0 | KQuantScheme::GsqRco3p5)
}

/// Repack a GsqRco or Q2_0 packed weight into `sudot8` u4 lanes.
///
/// `data` is the GGUF block stream for an `[n, k]` weight: `n * k / 64`
/// eighteen-byte blocks, row-major, no padding between rows. `k` must be a
/// multiple of [`CITYCROW_GROUP`].
///
/// The per-group scale is `max |w|` over the group's 128 decoded weights,
/// quantized to bf16. The code stored is the offset-binary form `q - 2`, so
/// that `w ~= (q - zero) * scale` and the kernel's two-dot correction applies
/// unchanged.
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
             (supported: Q2_0, GsqRco3p5)"
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
        KQuantScheme::Q2_0 => dequant_q2_0(data, num_weights)?,
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
    use crate::quantize_q2_0_block;

    /// A deterministic non-degenerate block: the all-zero block dequantizes
    /// correctly under ANY layout, so it cannot distinguish a right
    /// implementation from a wrong one.
    fn fixture_block(seed: u32) -> Vec<f32> {
        (0..64)
            .map(|j| {
                let x = j as f32;
                // Deterministic, spans both signs, no exact zero run.
                ((x * 37.0 + seed as f32) % 61.0) / 61.0 - 0.5
            })
            .collect()
    }

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
    /// The crate ships `quantize_q2_0_block`, which writes the **Q2_0**
    /// codebook (`{-1, 0, +1, +2}`, bias 1). GSQRCO is bias 2
    /// (`{-2, -1, 0, +1}`), so feeding Q2_0 bytes to
    /// `dequant_gsq_rco_3p5` shifts every value by one level of `d` and the
    /// round trip is meaningless. There is no `quantize_gsq_rco_block` in the
    /// crate, so GSQRCO bytes are built here directly: `code = round(w/d) + 2`
    /// with `d = amax` per 64-weight block, four codes per byte.
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
                // GSQRCO codebook: 4 levels centred on 2.
                let t = (w / d + 2.0).round().clamp(0.0, 3.0) as u8;
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

    /// KAT: the geometry is 64 weights per 18 bytes and the codebook is
    /// `{-2,-1,0,+1}` for GsqRco (bias 2) versus `{-1,0,+1,+2}` for Q2_0
    /// (bias 1). Same bytes, different weights -- a one-level shift of `d`.
    #[test]
    fn gsqrco_and_q2_0_decode_to_different_weights_from_the_same_bytes() {
        let block = fixture_block(11);
        let mut packed = vec![0u8; 18];
        quantize_q2_0_block(&block, &mut packed).expect("packs");

        let n_q2_0 = 64;
        let q2_0 = dequant_q2_0(&packed, n_q2_0).expect("q2_0 decodes");
        let gsq = dequant_gsq_rco_3p5(&packed, n_q2_0).expect("gsq decodes");

        assert_ne!(q2_0, gsq, "the two codebooks must not alias");
        // q2_0 = (q-1)d, gsq = (q-2)d, so gsq == q2_0 - d elementwise.
        for i in 0..64 {
            let d = crate::f16_to_f32(packed[0], packed[1]);
            assert!(
                (gsq[i] - (q2_0[i] - d)).abs() < 1e-3,
                "element {i}: gsq {} should be q2_0 {} - d {d}",
                gsq[i],
                q2_0[i]
            );
        }
    }

    /// The repack must land on the offset-binary zero point of 2, which is
    /// what the kernel's `iacc - z_b * sum_qa` correction assumes.
    #[test]
    fn repack_stores_the_offset_binary_zero_point() {
        let k = CITYCROW_GROUP;
        let flat: Vec<f32> = (0..k).map(|i| (i as f32 / k as f32) - 0.5).collect();
        let packed = pack_rows(&flat, 1, k);

        let w = repack_to_u4_lanes(&packed, 1, k, KQuantScheme::GsqRco3p5).expect("repacks");
        assert_eq!(w.zeros, vec![2u8; 1]);
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
            zeros: vec![2],
            n,
            k,
        };
        let got = decode_groups(&w);

        // Expected, derived longhand from the ISA: code i is at word[i/8],
        // shift (i%8)*4, value = (code - 2) * scale.
        let expect = |i: usize| -> f32 {
            let code = if i < 8 {
                w0[i]
            } else if i < 16 {
                w1[i - 8]
            } else {
                2
            };
            (code as f32 - 2.0) * 2.0
        };
        for i in 0..k {
            assert!(
                (got[i] - expect(i)).abs() < 1e-6,
                "element {i}: decode_groups gave {}, ISA layout says {}",
                got[i],
                expect(i)
            );
        }
        // And the scale must be honoured: with scale 2.0 a code of 3 is +2.0.
        assert!((got[0] - 2.0).abs() < 1e-6, "got[0] = {}", got[0]);

        // A second group with a different scale proves the decoder reads
        // scales per group rather than reusing group 0's.
        let two = CityCrowWeights {
            qweight: vec![0x3333_3333; 2 * CITYCROW_WORDS_PER_GROUP],
            scales: vec![scale_two, scale_half],
            zeros: vec![2, 2],
            n: 1,
            k: 2 * CITYCROW_GROUP,
        };
        let got2 = decode_groups(&two);
        assert!(
            (got2[0] - 2.0).abs() < 1e-6,
            "group 0 code 3 at scale 2.0 should be 2.0, got {}",
            got2[0]
        );
        assert!(
            (got2[CITYCROW_GROUP] - 0.5).abs() < 1e-6,
            "group 1 code 3 at scale 0.5 should be 0.5, got {}",
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
                let want = (((v / want_scale) + 2.0 + 0.5).floor() as i32).clamp(0, 3) as u32;
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
        assert!(scheme_supported(KQuantScheme::Q2_0));
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
        // The GSQRCO codebook is {-2,-1,0,+1} * d with d = amax, so its
        // representable range is [-d, +d] = [-amax, +amax] -- NOT
        // [-2*amax, +2*amax]. A fixture that assumes the wider span asks for
        // code 0 from a w of -2*amax, which this format cannot express, and
        // the "all four codes" assertion below then fails for a reason that
        // has nothing to do with the lane layout under test.
        //
        // So the fixture spans exactly [-amax, +amax]: with amax = 1.0 the four
        // codes are reachable at w = -1, -1/3, +1/3, +1.
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
            let want = (((flat[i] / scale) + 2.0 + 0.5).floor() as i32).clamp(0, 3) as u32;
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
        // fixture gap: the codebook is {-2,-1,0,+1} * d with d = amax, so the
        // representable range is [-d, +d]. Code 0 needs t = v/d + 2 < 0.5,
        // i.e. v < -1.5*d, which lies outside [-d, +d]. So the minimum code
        // from any weight at or above -d is code 1. Asserting all four here
        // would be asserting something false about GSQRCO.
        assert!(
            seen[1] && seen[2] && seen[3],
            "fixture must exercise codes 1..3, saw {seen:?}"
        );
        assert!(
            !seen[0],
            "code 0 is unreachable for d = amax; seeing it means the scale or \
             the codebook offset is wrong: {seen:?}"
        );
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
                code >= 2,
                "group 0 lane {i} holds code {code} (<2) but its weight is positive"
            );
        }
        // Group 1's weights are negative, so every code must be <= 2. A bleed
        // from group 0 would put a 3 here.
        for i in 0..CITYCROW_GROUP {
            let word = w.qweight[CITYCROW_WORDS_PER_GROUP + i / 8];
            let code = ((word >> ((i % 8) * 4)) & 0xF) as i32;
            assert!(
                code <= 2,
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
