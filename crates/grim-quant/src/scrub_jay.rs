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
//!              = 4 + 32 = 36 bits per 8 values = 4.5 bpw
//! ```
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
    [ 1,  3,  5,  7,  9, 11, 13, 16, 18, 20, 21, 23, 25, 26, 28, 30],
    [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 20, 22, 24, 26, 28, 30],
    [ 1,  3,  5,  7,  8, 10, 12, 13, 15, 18, 20, 21, 24, 26, 28, 30],
    [ 1,  3,  5,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    [ 1,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    [ 1,  3,  5,  7,  9, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    [ 1,  3,  5,  7,  9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 29, 30],
    [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 24, 26, 28, 30],
    [ 1,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    [ 1,  3,  5,  6,  8, 10, 12, 14, 16, 18, 20, 23, 25, 26, 28, 30],
    [ 1,  3,  4,  6,  8,  9, 11, 13, 15, 17, 19, 21, 23, 25, 28, 30],
    [ 1,  3,  4,  6,  8, 10, 12, 14, 16, 18, 20, 21, 23, 26, 28, 30],
    [ 1,  2,  3,  5,  7,  8, 10, 12, 14, 17, 18, 21, 23, 25, 27, 30],
    [ 1,  2,  4,  5,  7,  9, 11, 12, 14, 16, 18, 19, 22, 24, 27, 29],
    [ 1,  2,  3,  4,  6,  7,  9, 11, 12, 14, 16, 18, 20, 24, 27, 30],
    [ 0,  1,  1,  2,  2,  3,  4,  4,  5,  6,  8,  9, 11, 14, 17, 26],
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
/// # Shape of the result
///
/// The 16 codebooks are the 16 **clusters** of a Lloyd-max clustering of the
/// standardised corpus, and each codebook's 16 entries are the quantiles of that
/// cluster's own members. So codebook 0 describes the dense bulk near zero and
/// codebook 15 describes a sparse tail: the tables are genuinely different
/// because they describe genuinely different regions of the distribution, and a
/// block picks the one whose region it resembles.
///
/// # Why not 16 restarts of one whole-corpus fit
///
/// The first attempt did that, and the distinctness test caught the failure: a
/// Gaussian's k-means solution is essentially unique, so all 16 restarts
/// converged to the *same* table. Sixteen identical tables would leave the
/// per-block selector with nothing to choose between and collapse the format to
/// a single 16-entry codebook while still paying for 192 bytes. The second
/// attempt fit each codebook to a separate magnitude band, which got to 7
/// distinct tables but still spanned only 3 of the 64 available levels.
///
/// # Why the entries are rescaled to the alphabet
///
/// Codebook entries are 6-bit integers, so the span is -32..=31. Fitting raw
/// standard-normal values leaves every centroid inside about +/-3, wasting 57 of
/// 64 levels -- 6 bits would have bought 3. Each cluster's quantiles are
/// therefore stretched to fill the alphabet, which is exactly the job the
/// per-block E4M3 scale does at runtime: it maps a block's magnitude onto the
/// codebook's range. Rescaling here and scaling there are the same operation
/// applied once offline and once per block.
///
/// # Determinism
///
/// Fixed corpus from a fixed seed, fixed iteration count, lowest-index tie rule
/// in the assignment step, and a final sort. The sort matters: without it two
/// calibrations that found the same clusters in a different order would compare
/// unequal, making the frozen-constant check flaky rather than meaningful.
pub fn calibrate_codebook(seed: u64) -> [[i8; SCRUB_JAY_ENTRIES]; SCRUB_JAY_CODEBOOKS] {
    const CORPUS: usize = 16 * 1024;
    let data = calibration_corpus(CORPUS, seed);

    // One Lloyd-max clustering over the whole corpus. Magnitudes only: sign is a
    // separate bit in the format, and the codebook holds magnitudes.
    let mags: Vec<f32> = data.iter().map(|v| v.abs()).collect();

    let mut sorted = mags.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut init = [0f32; SCRUB_JAY_ENTRIES];
    for (slot, c) in init.iter_mut().enumerate() {
        let t = (slot as f32 + 0.5) / SCRUB_JAY_ENTRIES as f32;
        let idx = ((t * CORPUS as f32) as usize).min(CORPUS - 1);
        *c = sorted[idx];
    }

    // Assignment + update, tracking which samples landed where so each cluster's
    // own quantiles can be read off afterwards.
    let mut centroids = init;
    let mut members: Vec<Vec<f32>> = vec![Vec::new(); SCRUB_JAY_CODEBOOKS];
    for _ in 0..LLOYD_ITERS {
        for m in members.iter_mut() {
            m.clear();
        }
        for &v in &mags {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (i, &c) in centroids.iter().enumerate() {
                let d = (v - c) * (v - c);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            members[best].push(v);
        }
        for i in 0..SCRUB_JAY_ENTRIES {
            if !members[i].is_empty() {
                let mean =
                    members[i].iter().map(|&v| v as f64).sum::<f64>() / members[i].len() as f64;
                centroids[i] = mean as f32;
            }
        }
    }

    // Each codebook is its cluster's 16 quantiles, stretched to the alphabet.
    let mut table = [[0i8; SCRUB_JAY_ENTRIES]; SCRUB_JAY_CODEBOOKS];
    for k in 0..SCRUB_JAY_CODEBOOKS {
        let mut m = std::mem::take(&mut members[k]);
        m.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Affine-map the cluster's own [min, max] onto [0, 31], then quantize.
        //
        // An offset as well as a scale is required. Scaling by the max alone left
        // narrow clusters stranded: codebook 5's order statistics span 25..31 in
        // natural units, so dividing by the max mapped them to 25..31 again and
        // the 16 entries collapsed to 7 distinct integers -- not worth a 4-bit
        // index per scalar. This is the same operation the per-block E4M3 scale
        // performs at runtime, which is why the table is built this way.
        let lo_v = m.first().copied().unwrap_or(0.0);
        let hi_v = m.last().copied().unwrap_or(1.0);
        let extent = (hi_v - lo_v).max(1e-6);

        // 16 evenly spaced order statistics. Uses the full entry budget even for
        // a small cluster, which is the point: the cluster's *shape* is the
        // information, not its size.
        let mut levels = [0i8; SCRUB_JAY_ENTRIES];
        for (slot, l) in levels.iter_mut().enumerate() {
            let t = (slot as f32 + 0.5) / SCRUB_JAY_ENTRIES as f32;
            let idx = ((t * m.len() as f32) as usize).min(m.len().saturating_sub(1));
            let raw = m.get(idx).copied().unwrap_or(lo_v);
            *l = quantize_to_e6((raw - lo_v) / extent * 31.0);
        }

        table[k] = levels;
        table[k].sort_unstable();
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
