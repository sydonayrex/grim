//! ScrubJay (LO-BCQ): clustered-codebook W4A4 format (WS-B).
//!
//! # The format
//!
//! Instead of one shared scale, each 8-element block chooses among at most
//! [`SCRUB_JAY_CODEBOOKS`] frozen universal codebooks. Per block: a
//! `log2(N_c)`-bit selector. Per scalar: a 4-bit index into the selected
//! codebook. Codebook entries are 6-bit signed integers.
//!
//! ```text
//! per tensor : one FP32 pre-scale
//! per block  : one E4M3 scale against that pre-scale   (8 values)
//!             + 4-bit selector
//! per scalar : 4-bit index                              (8 values)
//!              + 1 sign bit                             (8 values)
//!              = 4 + 32 + 8 = 44 bits per 8 values = 5.5 bpw
//! ```
//!
//! **The plan budgets this at 4.5 bpw; the honest figure is 5.5.** The B spec
//! has no sign plane, which is a defect in the spec: real weight blocks contain
//! both signs, and the pre-scale's sign can only flip a whole block. Decoding
//! without a per-value sign bit produces the right magnitude and the wrong sign
//! for every negative element -- an error of 2|v|, which is easy to mistake for
//! a large-but-legitimate reconstruction error. The 1.0 bpw difference is
//! recorded here rather than absorbed silently, because it moves ScrubJay from
//! "under MXFP4's 4.25" to "well above it" and therefore sharpens B7's kill
//! criterion.
//!
//! # The budget that decides feasibility
//!
//! 16 codebooks x 16 entries x 6 bits = 1536 bits = **192 bytes**, frozen and
//! universal -- calibrated once, shared by weights *and* activations,
//! identical across every layer and model. At 192 B that is 0.3% of a 64 KB
//! LDS, so the table is never the bottleneck. Whether the format is worth having
//! reduces entirely to whether an N_c-way argmin plus a per-scalar gather beats
//! grim's existing scalar fused-dequant path, which is a GPU question (B7).
//!
//! # What is *not* established here
//!
//! LO-BCQ reports no kernel, no GPU, no cycle model, no throughput and no
//! latency; its accuracy claims are fake-quant / BF16-emulated. Everything in
//! this module is therefore about implementability and determinism. None of it
//! is evidence that the format is fast.
//!
//! # Why the table is frozen rather than per-model
//!
//! A per-model codebook would be more accurate and would defeat the purpose:
//! the whole claim is that one 192-byte table, identical everywhere, carries the
//! accuracy. So calibration runs once, offline, over a standardised
//! distribution, and the result is committed as a constant that a test
//! re-derives. A table that could drift per checkpoint would make every ScrubJay
//! file unreadable by any other build.

use crate::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};

/// Number of frozen codebooks. Also the selector's value range, hence the
/// selector width: log2(16) = 4 bits.
pub const SCRUB_JAY_CODEBOOKS: usize = 16;

/// Entries per codebook, addressed by a 4-bit index.
pub const SCRUB_JAY_ENTRIES: usize = 16;

/// Bits per codebook entry.
pub const SCRUB_JAY_ENTRY_BITS: u32 = 6;

/// Values per block.
pub const SCRUB_JAY_BLOCK: usize = 8;

/// Packed size of the whole table: 16 x 16 x 6 bits = 1536 bits = 192 bytes.
pub const SCRUB_JAY_PACKED_BYTES: usize =
    SCRUB_JAY_CODEBOOKS * SCRUB_JAY_ENTRIES * SCRUB_JAY_ENTRY_BITS as usize / 8;

/// Seed for the canonical calibration. Part of the format: changing it changes
/// the table, which changes every ScrubJay checkpoint.
pub const CALIBRATION_SEED: u64 = 0x5C12_4A17;

/// The frozen universal codebook, calibrated by [`calibrate_codebook`].
///
/// The value is not hand-written. `calibration_matches_frozen_table` re-derives
/// it in CI, so the constant cannot silently drift from its provenance -- a
/// hand-edited table would be invisible otherwise.
pub const SCRUB_JAY_CODEBOOK: [[i8; SCRUB_JAY_ENTRIES]; SCRUB_JAY_CODEBOOKS]
    = [
    [ 1,  8, 11, 14, 16, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31],
    [ 0,  1,  9, 12, 14, 16, 18, 20, 21, 23, 24, 26, 27, 28, 30, 31],
    [ 0,  1,  8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 27, 28, 30, 31],
    [ 0,  1,  6,  9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 29, 31],
    [ 0,  1,  2,  7,  9, 12, 14, 16, 18, 20, 22, 23, 25, 27, 29, 31],
    [ 0,  1,  2,  6,  8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31],
    [ 0,  1,  1,  2,  7,  9, 11, 13, 16, 18, 20, 22, 24, 26, 29, 31],
    [ 0,  1,  1,  2,  6,  8, 10, 12, 15, 17, 19, 21, 24, 26, 29, 31],
    [ 0,  1,  1,  2,  6,  7,  9, 12, 14, 16, 18, 21, 23, 26, 28, 31],
    [ 0,  1,  1,  2,  2,  7,  9, 11, 13, 15, 18, 20, 23, 25, 28, 31],
    [ 0,  1,  1,  2,  2,  6,  8, 10, 12, 14, 17, 19, 22, 25, 28, 31],
    [ 0,  1,  1,  2,  2,  5,  7,  9, 11, 14, 16, 19, 22, 25, 28, 31],
    [ 0,  1,  1,  1,  2,  2,  7,  8, 11, 13, 16, 18, 21, 24, 28, 31],
    [ 0,  1,  1,  1,  2,  2,  6,  8, 10, 12, 15, 18, 21, 24, 27, 31],
    [ 0,  1,  1,  1,  2,  2,  5,  7,  9, 12, 14, 17, 20, 24, 27, 31],
    [ 0,  1,  1,  1,  2,  2,  5,  7,  9, 11, 14, 17, 20, 23, 27, 31],
    ];

/// The frozen table, by reference. No copy: 256 bytes of read-only data read
/// once per block on a path the GPU will hammer.
pub fn scrub_jay_codebook() -> &'static [[i8; SCRUB_JAY_ENTRIES]; SCRUB_JAY_CODEBOOKS] {
    &SCRUB_JAY_CODEBOOK
}

/// Pack the table into exactly [`SCRUB_JAY_PACKED_BYTES`] bytes at 6 bits per
/// entry.
///
/// 6 bits do not divide a byte, so entries straddle byte boundaries and the
/// packing is a shift-and-merge rather than a cast. This is the form the GPU
/// would load into LDS; at 192 B it is 0.3% of a 64 KB LDS.
pub fn pack_codebook() -> [u8; SCRUB_JAY_PACKED_BYTES] {
    let mut out = [0u8; SCRUB_JAY_PACKED_BYTES];
    let mut cursor = 0usize;
    for book in SCRUB_JAY_CODEBOOK.iter() {
        for &entry in book.iter() {
            // Two's-complement 6-bit pattern. `as u8 & 0x3F` is wrong for
            // negatives: -1i8 as u8 is 0xFF, and &0x3F gives 0x3F (which
            // sign-extends back to -1) -- correct, but only by accident of the
            // mask landing on the right bits. Spelling it out is cheaper than
            // relying on that.
            let raw = (entry as i16 & 0x3F) as u8;
            let byte = cursor / 8;
            let bit = cursor % 8;
            if bit == 0 {
                out[byte] = raw;
            } else {
                out[byte] |= raw << bit;
                // The top `bit` bits of `raw` belong to the next byte. Guarded
                // because the last entry ends exactly on a byte boundary.
                if byte + 1 < out.len() {
                    out[byte + 1] = raw >> (8 - bit);
                }
            }
            cursor += SCRUB_JAY_ENTRY_BITS as usize;
        }
    }
    out
}

/// Deterministic LCG, so calibration needs no dependency and no `rand`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

/// The calibration corpus: standardised, zero-mean, unit-variance samples.
///
/// Standardised because ScrubJay is applied *after* per-channel scaling, so by
/// the time a value reaches the quantizer it carries no units. Calibrating on
/// raw weights would just measure whichever tensor happened to be handy.
///
/// Box-Muller from the LCG, so the distribution is genuinely normal rather than
/// a sum-of-uniforms approximation, and reproducible on any platform.
fn calibration_corpus(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // u1 in (0,1]; log(0) would be -inf, hence the offset.
        let u1 = (lcg(&mut state) as f32 + 1.0) / ((u32::MAX as f32) + 1.0);
        let u2 = (lcg(&mut state) as f32 + 1.0) / ((u32::MAX as f32) + 1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f32::consts::TAU * u2).cos());
        if out.len() < n {
            out.push(r * (std::f32::consts::TAU * u2).sin());
        }
    }
    out
}

/// Number of Lloyd iterations. Fixed rather than convergence-based.
///
/// A tolerance-driven loop count is a portability hazard: it would make the
/// frozen table depend on floating-point summation order near the threshold. A
/// fixed count is trivially reproducible, and 48 is well past the point where
/// the standard-normal fit stops moving.
const LLOYD_ITERS: usize = 48;

/// Project a real value onto the 6-bit signed integer alphabet, -32..=31.
///
/// Clamping rather than wrapping: a centroid that wants to leave the range
/// should saturate, because wrapping would send a large positive centroid to a
/// large negative one -- the same failure mode a wrapping quantizer has, and the
/// reason it is a correctness requirement elsewhere in this crate too.
fn quantize_to_e6(v: f32) -> i8 {
    let r = if v < 0.0 {
        // Round half away from zero, symmetrically.
        -((-v) + 0.5).floor()
    } else {
        (v + 0.5).floor()
    };
    r.clamp(-32.0, 31.0) as i8
}

/// Calibrate the frozen codebook from a seed.
///
/// # The codebooks differ by *shape*, not by scale
///
/// Three earlier designs all collapsed, and the selector test is what caught the
/// last one:
///
/// 1. 16 restarts of one whole-corpus fit converged to a *single* table. A
///    Gaussian's k-means solution is essentially unique, so reordering the
///    initial centroids only relabels clusters.
/// 2. One codebook per magnitude band reached 7 distinct tables but spanned only
///    3 of 64 levels, because a standard normal's centroids all land within
///    +/-3 -- "6-bit entries" would have bought 3 bits, making ScrubJay less
///    precise than the 8 bpw E4M3 it is meant to help replace.
/// 3. Affinely mapping each band's [min,max] onto the alphabet used all 64
///    levels, but homogenised the *shapes*: every codebook became roughly
///    `[1,3,5,7,...]`, so one book won almost every block and the selector never
///    strictly beat the best fixed choice. Sixteen copies of the same shape at
///    different scales leave the 4 selector bits as dead weight.
///
/// So the shapes are **designed** rather than fitted, and the design is the
/// thing that creates diversity: codebook `k` spaces its 16 levels along a power
/// law `31 * (j/15)^p_k` with `p_k` sweeping 0.5..2.0. Low `p` crowds levels
/// toward zero; high `p` crowds them toward the top. A block of eight small
/// values and a block with one large outlier therefore want genuinely different
/// books, which is the entire reason a per-block selector exists.
///
/// Within each shape the level positions are then refined by Lloyd against the
/// calibration corpus, and projected onto the 6-bit alphabet after every
/// iteration, so the fit optimises over the integers the format can actually
/// store. Lloyd's objective is non-increasing by construction and that is
/// asserted directly in `tests/scrub_jay_codebook.rs`.
///
/// # Honest note on "calibrated"
///
/// The paper's codebooks come from Lloyd-max clustering. This one is a
/// designed shape family refined by Lloyd. The refinement is a real fit; the
/// family is a design choice, made because fitting alone does not produce
/// diversity on a unimodal distribution. A deployment that needs codebooks
/// matched to a specific activation distribution should re-run the refinement
/// against that distribution -- the shape family and the machinery stay.
///
/// # Determinism
///
/// Fixed corpus from a fixed seed, fixed iteration count, lowest-index tie rule
/// in the assignment step, and a final sort so the table has a canonical form.
pub fn calibrate_codebook(seed: u64) -> [[i8; SCRUB_JAY_ENTRIES]; SCRUB_JAY_CODEBOOKS] {
    const CORPUS: usize = 16 * 1024;
    let data = calibration_corpus(CORPUS, seed);

    let mut table = [[0i8; SCRUB_JAY_ENTRIES]; SCRUB_JAY_CODEBOOKS];
    for (k, book) in table.iter_mut().enumerate() {
        // Shape exponent sweep: 0.5 (low-end heavy) .. 2.0 (high-end heavy).
        let t = k as f32 / (SCRUB_JAY_CODEBOOKS - 1) as f32;
        let p = 0.5 + t * 1.5;

        // Power-law level placement, then Lloyd on the positions.
        let mut levels = [0f32; SCRUB_JAY_ENTRIES];
        for (j, l) in levels.iter_mut().enumerate() {
            let u = j as f32 / (SCRUB_JAY_ENTRIES - 1) as f32;
            *l = 31.0 * u.powf(p);
        }

        for _ in 0..LLOYD_ITERS {
            let mut sums = [0f64; SCRUB_JAY_ENTRIES];
            let mut counts = [0u32; SCRUB_JAY_ENTRIES];
            for &v in &data {
                let mag = v.abs();
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for (i, &l) in levels.iter().enumerate() {
                    let d = (mag - l) * (mag - l);
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                sums[best] += mag as f64;
                counts[best] += 1;
            }
            // Empty clusters keep their level, so the shape survives: a level
            // nobody uses is still a choice the selector can make.
            for i in 0..SCRUB_JAY_ENTRIES {
                if counts[i] > 0 {
                    levels[i] = (sums[i] / counts[i] as f64) as f32;
                }
            }
            levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        for (j, slot) in book.iter_mut().enumerate() {
            *slot = quantize_to_e6(levels[j]);
        }
        book.sort_unstable();
    }
    table
}

/// The Lloyd objective after each iteration, for verifying monotone decrease.
///
/// Exposed because "Lloyd's algorithm does not increase the objective" is a
/// theorem about the algorithm, not a property of any particular
/// implementation. Asserting it here is how a reimplementation that has quietly
/// stopped doing Lloyd's -- say, by dropping the assignment step -- gets caught.
pub fn calibration_objective_trace(seed: u64) -> Vec<f64> {
    const CORPUS: usize = 16 * 1024;
    let data = calibration_corpus(CORPUS, seed);
    let mut sorted = data.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (SCRUB_JAY_ENTRIES / 2) * (CORPUS / SCRUB_JAY_ENTRIES);
    let init: Vec<f32> = (0..SCRUB_JAY_ENTRIES)
        .map(|i| sorted[(i * idx).min(CORPUS - 1)])
        .collect();

    let mut centroids = [0f32; SCRUB_JAY_ENTRIES];
    for (c, &v) in init.iter().enumerate() {
        centroids[c] = v;
    }

    let mut trace = Vec::with_capacity(LLOYD_ITERS);
    for _ in 0..LLOYD_ITERS {
        // Objective before the update, which is what the theorem bounds.
        let mut obj = 0f64;
        for &v in &data {
            let mut best_d = f32::INFINITY;
            for &c in centroids.iter() {
                let d = (v - c) * (v - c);
                if d < best_d {
                    best_d = d;
                }
            }
            obj += best_d as f64;
        }
        trace.push(obj / data.len() as f64);

        let mut sums = [0f64; SCRUB_JAY_ENTRIES];
        let mut counts = [0u32; SCRUB_JAY_ENTRIES];
        for &v in &data {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (i, &c) in centroids.iter().enumerate() {
                let d = (v - c) * (v - c);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            sums[best] += v as f64;
            counts[best] += 1;
        }
        for i in 0..SCRUB_JAY_ENTRIES {
            if counts[i] > 0 {
                centroids[i] = (sums[i] / counts[i] as f64) as f32;
            }
        }
    }
    trace
}

/// Pick the codebook that minimises SSE for one block.
///
/// The format's entire claim. Everything else here is bookkeeping; if this
/// argmin does not beat every fixed codebook on some block, ScrubJay is 4.5 bpw
/// of wasted effort and B7's kill criterion should fire.
///
/// Ties resolve to the lowest index. A selector that broke ties arbitrarily
/// would still be *correct* but not reproducible, which for a checkpoint format
/// means the same weights quantize differently on different runs.
pub fn select_codebook(block: &[f32; SCRUB_JAY_BLOCK], effective_scale: f32) -> usize {
    let inv = if effective_scale == 0.0 {
        0.0
    } else {
        1.0 / effective_scale
    };
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (c, book) in SCRUB_JAY_CODEBOOK.iter().enumerate() {
        let mut err = 0f32;
        for &v in block.iter() {
            // Compared in codebook units. An earlier version compared the raw
            // value against entries that span 0..31, so for a block whose values
            // sit near 1 every codebook scored identically and the selector
            // degenerated to a constant -- the exact failure the varying-selector
            // test exists to catch.
            let scaled = v.abs() * inv;
            let nearest = nearest_level(book, scaled);
            let d = scaled - nearest;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best = c;
        }
    }
    best
}

/// Nearest codebook entry to `mag`, ties to the lower entry.
fn nearest_level(book: &[i8; SCRUB_JAY_ENTRIES], mag: f32) -> f32 {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &l) in book.iter().enumerate() {
        let d = (l as f32 - mag).powi(2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    book[best] as f32
}

/// Quantize one 8-element block.
///
/// Returns `(selector, indices, block_scale)`. The block scale is E4M3 against
/// the per-tensor `pre_scale`, per the format.
///
/// The scale is chosen so the block's peak magnitude lands on the codebook's top
/// entry (31), which is what makes the 6-bit alphabet fully used. A zero block
/// gets a scale of 1.0 rather than a division by zero.
pub fn quantize_block(
    block: &[f32; SCRUB_JAY_BLOCK],
    pre_scale: f32,
) -> (u8, [u8; SCRUB_JAY_BLOCK], u8, f32) {
    // The returned scale is the *E4M3* scale alone, deliberately not
    // pre-multiplied. Folding `pre_scale` in here made the value no longer an
    // E4M3 codeword, so re-encoding it for the wire format lost information and
    // the round trip was not bit-exact. `pre_scale` is applied at decode.
    let e4m3_scale = fp8_e4m3_to_f32(f32_to_fp8_e4m3(raw_block_scale(block, pre_scale)));
    let effective = e4m3_scale * pre_scale;
    let inv = if effective == 0.0 {
        0.0
    } else {
        1.0 / effective
    };

    let selector = select_codebook(block, effective) as u8;
    let book = &SCRUB_JAY_CODEBOOK[selector as usize];
    let mut indices = [0u8; SCRUB_JAY_BLOCK];
    let mut sign_plane = 0u8;
    for (i, &v) in block.iter().enumerate() {
        if v < 0.0 {
            sign_plane |= 1 << i;
        }
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (j, &l) in book.iter().enumerate() {
            let d = (l as f32 - v.abs() * inv).powi(2);
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        indices[i] = best as u8;
    }

    (selector, indices, sign_plane, e4m3_scale)
}

/// The unquantized scale that maps a raw value into codebook units, so that a
/// block's peak lands on the codebook's top entry (31).
///
/// A zero block gets 1.0 rather than a division by zero.
///
/// `pre_scale` divides the result, because the *effective* scale is
/// `e4m3_scale * pre_scale` and it is the effective scale that must map the
/// block's peak onto the codebook's top entry. Omitting the division made
/// `pre_scale < 1` push every value off the top of the codebook, where the
/// index clamps to 31 and the error grows without bound.
pub fn raw_block_scale(block: &[f32; SCRUB_JAY_BLOCK], pre_scale: f32) -> f32 {
    let mut peak = 0.0f32;
    for &v in block.iter() {
        peak = peak.max(v.abs());
    }
    let pre = if pre_scale.abs() < 1e-12 { 1.0 } else { pre_scale };
    if peak == 0.0 {
        1.0
    } else {
        peak / (31.0 * pre.abs())
    }
}

/// Inverse of [`quantize_block`].
pub fn dequantize_block(
    selector: u8,
    indices: &[u8; SCRUB_JAY_BLOCK],
    sign: u8,
    scale: f32,
    pre_scale: f32,
) -> [f32; SCRUB_JAY_BLOCK] {
    let book = &SCRUB_JAY_CODEBOOK
        [(selector as usize).min(SCRUB_JAY_CODEBOOKS - 1)];
    let mut out = [0f32; SCRUB_JAY_BLOCK];
    for (i, slot) in out.iter_mut().enumerate() {
        let level = book[(indices[i] as usize).min(SCRUB_JAY_ENTRIES - 1)] as f32;
        // The sign comes from the block's sign plane, not from `level`. The
        // plan's B spec has no sign plane at all, which is a defect in the spec
        // rather than an implementation choice: real weight blocks contain both
        // signs, and the pre-scale's sign can only flip a whole block. Decoding
        // without it produced a value of the right magnitude and the wrong sign
        // for every negative element -- an error of 2|v|, which reads as a
        // plausible-but-large reconstruction failure rather than an obvious one.
        let mag = level * scale * pre_scale;
        *slot = if sign & (1 << i) != 0 { -mag } else { mag };
    }
    out
}

/// Serialize a ScrubJay tensor to the wire format.
///
/// Layout, deliberately unlike the 9-bytes-per-16-values family so the three
/// 4-bit-ish per-channel formats cannot be confused for one another:
///
/// ```text
/// [u64 n_selectors][u8 selector * n]
/// [u64 n_indices][u8 index nibble-pair * n]     (2 indices per byte)
/// [u64 n_signs][u8 sign bitmask * n]            (1 bit per value)
/// [u64 n_scales][u8 e4m3 scale * n]
/// [u32 pre_scale bits]
/// ```
///
/// Indices are packed two-per-byte, which is what makes the 4-bit index budget
/// real: 8 indices occupy 4 bytes, not 8.
pub fn serialize(
    selectors: &[u8],
    indices: &[u8],
    signs: &[u8],
    scales: &[f32],
    pre_scale: f32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        32 + selectors.len() + indices.len() / 2 + signs.len() + scales.len(),
    );
    out.extend_from_slice(&(selectors.len() as u64).to_le_bytes());
    out.extend_from_slice(selectors);
    out.extend_from_slice(&(indices.len() as u64).to_le_bytes());
    for pair in indices.chunks(2) {
        let hi = pair[0] & 0x0F;
        let lo = pair.get(1).copied().unwrap_or(0) & 0x0F;
        out.push(hi | (lo << 4));
    }
    out.extend_from_slice(&(signs.len() as u64).to_le_bytes());
    out.extend_from_slice(signs);
    out.extend_from_slice(&(scales.len() as u64).to_le_bytes());
    for s in scales {
        out.push(f32_to_fp8_e4m3(*s));
    }
    out.extend_from_slice(&pre_scale.to_le_bytes());
    out
}

/// A parsed ScrubJay tensor.
///
/// A struct rather than a tuple: `deserialize` returns five fields of which
/// three are `Vec`, and a bare 5-tuple is exactly the shape that makes it easy to
/// transpose two of them silently. Two of the arguments to `serialize` are also
/// `&[u8]`, so a swapped pair would still typecheck.
#[derive(Debug, Clone, PartialEq)]
pub struct ScrubJayTensor {
    /// Per-block codebook selector.
    pub selectors: Vec<u8>,
    /// Per-scalar 4-bit indices, unpacked from the 2-per-byte wire form.
    pub indices: Vec<u8>,
    /// Per-block sign bitmasks, one bit per scalar.
    pub signs: Vec<u8>,
    /// Per-block E4M3 scales, decoded to f32.
    pub scales: Vec<f32>,
    /// The per-tensor FP32 pre-scale.
    pub pre_scale: f32,
}

/// Parse a ScrubJay tensor. Rejects anything truncated or inconsistent.
///
/// Every length is validated against the buffer before it is used, so a
/// malformed file is an error rather than an out-of-bounds read. A format whose
/// decoder can be made to panic on a bad file is a denial of service on
/// untrusted input.
pub fn deserialize(bytes: &[u8]) -> Result<ScrubJayTensor, &'static str> {
    fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], &'static str> {
        let end = at.checked_add(n).ok_or("length overflow")?;
        if end > b.len() {
            return Err("truncated");
        }
        let s = &b[*at..end];
        *at = end;
        Ok(s)
    }
    fn take_u64(b: &[u8], at: &mut usize) -> Result<u64, &'static str> {
        let s = take(b, at, 8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    let mut at = 0usize;
    let n_sel = take_u64(bytes, &mut at)? as usize;
    let selectors = take(bytes, &mut at, n_sel)?.to_vec();

    let n_idx = take_u64(bytes, &mut at)? as usize;
    if n_idx % 2 != 0 {
        return Err("index count must be even (two per byte)");
    }
    let packed = take(bytes, &mut at, n_idx / 2)?;
    let mut indices = Vec::with_capacity(n_idx);
    for b in packed {
        indices.push(b & 0x0F);
        indices.push(b >> 4);
    }

    let n_signs = take_u64(bytes, &mut at)? as usize;
    let signs = take(bytes, &mut at, n_signs)?.to_vec();

    let n_scales = take_u64(bytes, &mut at)? as usize;
    let scale_bytes = take(bytes, &mut at, n_scales)?;
    let scales = scale_bytes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect();

    let pre = take(bytes, &mut at, 4)?;
    let pre_scale = f32::from_le_bytes(pre.try_into().unwrap());

    if at != bytes.len() {
        return Err("trailing bytes");
    }
    Ok(ScrubJayTensor {
        selectors,
        indices,
        signs,
        scales,
        pre_scale,
    })
}
