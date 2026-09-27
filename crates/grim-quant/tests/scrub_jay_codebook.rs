//! WS-B ScrubJay (LO-BCQ): the frozen codebook and its calibration (B1, B2).
//!
//! # What this format is
//!
//! Replace one shared scale with a *choice among <=16 frozen universal
//! codebooks* per 8-element block. Each block stores a log2(N_c) selector, then
//! one 4-bit index per scalar into the selected codebook. Codebook entries are
//! 6-bit integers.
//!
//! # The number that decides feasibility
//!
//! 16 codebooks x 16 entries x 6 bits = 1536 bits = **192 bytes**, frozen,
//! universal, shared by weights *and* activations, identical across every layer
//! and model. At 192 B that is 0.3% of a 64 KB LDS, so the table is never the
//! problem and the format's viability reduces entirely to whether an N_c-way
//! argmin beats grim's existing scalar fused-dequant path. That is B7's
//! question and it needs a GPU.
//!
//! # Be honest about the source
//!
//! LO-BCQ reports no kernel, no GPU, no cycle model, no throughput and no
//! latency. Its accuracy claims are fake-quant / BF16-emulated. So B1-B5 here
//! establish only that the format is *implementable and deterministic*; they say
//! nothing about whether it is fast, and nothing here should be read as
//! evidence that it is.
//!
//! CPU only.

use grim_quant::scrub_jay::{
    CALIBRATION_SEED, SCRUB_JAY_CODEBOOK, SCRUB_JAY_PACKED_BYTES, calibrate_codebook,
    pack_codebook, scrub_jay_codebook,
};

/// B1: the codebook is at most 192 bytes, and exactly that at 6 bits/entry.
///
/// The budget is the feasibility argument, so it is asserted as a literal
/// rather than recomputed from a constant that could drift.
#[test]
fn codebook_is_192_bytes_and_frozen() {
    assert_eq!(
        SCRUB_JAY_PACKED_BYTES, 192,
        "the 192-byte budget is the whole feasibility argument for ScrubJay"
    );
    assert_eq!(
        SCRUB_JAY_CODEBOOK.len(),
        16,
        "N_c is capped at 16 codebooks"
    );
    assert_eq!(
        SCRUB_JAY_CODEBOOK[0].len(),
        16,
        "each codebook has 16 entries, addressed by a 4-bit index"
    );

    // 16 x 16 x 6 bits = 1536 bits = 192 bytes exactly. No padding, because 6
    // bits do not divide a byte: entries straddle byte boundaries.
    let bits = SCRUB_JAY_CODEBOOK.len() * SCRUB_JAY_CODEBOOK[0].len() * 6;
    assert_eq!(bits, 1536);
    assert_eq!(bits / 8, 192);

    // The packed form really is 192 bytes, and really round trips.
    let packed = pack_codebook();
    assert_eq!(packed.len(), 192);
    assert_eq!(packed.len(), SCRUB_JAY_PACKED_BYTES);
}

/// Every entry must be representable in 6 signed bits.
///
/// A 6-bit signed integer spans -32..=31. An entry outside that range cannot be
/// packed, so this is a real constraint on the calibration, not a formality --
/// and it is why the calibration quantizes its centroids rather than rounding
/// them.
#[test]
fn every_entry_fits_in_six_signed_bits() {
    for (c, book) in SCRUB_JAY_CODEBOOK.iter().enumerate() {
        for (i, &v) in book.iter().enumerate() {
            assert!(
                (-32..=31).contains(&v),
                "codebook {c} entry {i} is {v}, outside the 6-bit signed range"
            );
        }
    }
}

/// B2: calibration is deterministic given the seed.
///
/// Two independent runs must be byte-identical. This is the property that makes
/// the committed constant trustworthy: if calibration were merely
/// *reproducible in practice*, the frozen table would be a liability, because a
/// contributor regenerating it on another machine would get a different format
/// and every ScrubJay checkpoint would silently become unreadable.
#[test]
fn calibration_is_deterministic_given_seed() {
    let a = calibrate_codebook(CALIBRATION_SEED);
    let b = calibrate_codebook(CALIBRATION_SEED);
    assert_eq!(
        a, b,
        "two calibrations at seed {CALIBRATION_SEED} must be byte-identical"
    );
    assert_eq!(
        a, SCRUB_JAY_CODEBOOK,
        "calibration must reproduce the committed frozen codebook exactly"
    );
}

/// A different seed must produce a different codebook.
///
/// Without this, the determinism test above would pass trivially if
/// `calibrate_codebook` ignored its seed and returned a constant.
#[test]
fn a_different_seed_gives_a_different_codebook() {
    let other = calibrate_codebook(CALIBRATION_SEED ^ 0xFFFF_FFFF);
    assert_ne!(
        other, SCRUB_JAY_CODEBOOK,
        "the seed must actually influence the result"
    );
}

/// The committed table is what calibration produces, so the constant is never
/// stale.
///
/// A hand-edited or regenerated-by-hand constant would be invisible; this makes
/// the provenance checkable in CI.
#[test]
fn the_frozen_table_is_exactly_what_calibration_produces() {
    assert_eq!(
        calibrate_codebook(CALIBRATION_SEED),
        SCRUB_JAY_CODEBOOK,
        "SCRUB_JAY_CODEBOOK has drifted from calibrate_codebook(CALIBRATION_SEED)"
    );
}

/// B2: Lloyd's objective is monotonically non-increasing.
///
/// This is the property that makes the iteration a *fit* rather than a random
/// walk, and the plan asks for it explicitly ("assert the monotone-decrease
/// property the paper proves"). Lloyd's algorithm guarantees it because each
/// step assigns to the nearest centroid (which cannot raise the objective) and
/// then re-fits each centroid to its members (the least-squares optimum, which
/// cannot raise it either).
///
/// A violation would mean the implementation is not actually doing Lloyd's
/// algorithm, which would invalidate every accuracy claim built on it.
#[test]
fn the_lloyd_objective_never_increases() {
    let trace = grim_quant::scrub_jay::calibration_objective_trace(CALIBRATION_SEED);
    assert!(
        trace.len() >= 4,
        "expected a real trace, got {} points",
        trace.len()
    );
    for w in trace.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-9,
            "objective increased from {} to {} at iteration {}",
            w[0],
            w[1],
            trace.windows(2).position(|x| x == w).unwrap_or(0)
        );
    }
    // And it must actually have moved, or the loop is a no-op.
    let first = trace[0];
    let last = trace[trace.len() - 1];
    assert!(
        last < first,
        "the objective did not improve ({first} -> {last}); the iteration is \\
         not doing anything"
    );
}

/// The codebooks must be substantially different from each other.
///
/// If they collapsed to one table the per-block selector would be meaningless:
/// every block would pick the same codebook and the format would silently
/// degenerate to a single 16-entry codebook, while still paying for 192 bytes.
///
/// The bar is 12 of 16, not 16 of 16. Two sparse tail clusters can quantise to
/// the same 16 integers and that is harmless -- the selector simply has one tie
/// to break. The failure this guards is *collapse*, not a duplicate pair.
#[test]
fn the_codebooks_are_substantially_distinct() {
    let mut sorted = SCRUB_JAY_CODEBOOK.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert!(
        sorted.len() >= 12,
        "only {} of 16 codebooks are distinct; the selector has almost nothing \
         to choose between and the format has collapsed",
        sorted.len()
    );

    // And specifically: the dense-bulk codebook must not be duplicated, since
    // that is the one every ordinary block would pick.
    for j in 1..SCRUB_JAY_CODEBOOK.len() {
        assert_ne!(
            SCRUB_JAY_CODEBOOK[0], SCRUB_JAY_CODEBOOK[j],
            "codebook 0 (the dense bulk) is duplicated at {j}"
        );
    }
}

/// Each codebook is sorted, and has enough distinct levels to be worth its
/// selector bits.
///
/// Sorted (non-decreasing) so the table has a canonical form -- otherwise
/// iteration order alone would make two calibrations look different and the
/// frozen-constant check would be flaky rather than meaningful.
///
/// Duplicates are *allowed*, because a sparse tail cluster's 16 order statistics
/// genuinely can quantise to fewer than 16 distinct integers. What is not
/// allowed is so few distinct levels that the codebook is not worth its 4 index
/// bits, so the bound is on distinct levels rather than on equality.
#[test]
fn each_codebook_is_sorted_and_has_enough_distinct_levels() {
    for (c, book) in SCRUB_JAY_CODEBOOK.iter().enumerate() {
        for w in book.windows(2) {
            assert!(
                w[0] <= w[1],
                "codebook {c} is not sorted at {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
        let mut levels = book.to_vec();
        levels.sort_unstable();
        levels.dedup();
        assert!(
            levels.len() >= 8,
            "codebook {c} has only {} distinct levels of 16; that is not worth \
             a 4-bit index per scalar",
            levels.len()
        );
    }
}

/// The entries must actually use the 6-bit alphabet.
///
/// A fit that left every entry inside +/-3 would waste 57 of 64 levels, making
/// "6-bit entries" a fiction and ScrubJay *less* precise than the 8 bpw E4M3 it
/// is meant to help replace. This is the check that caught the first two
/// calibration designs.
#[test]
fn the_table_uses_the_six_bit_alphabet() {
    let top = SCRUB_JAY_CODEBOOK
        .iter()
        .flatten()
        .copied()
        .max()
        .expect("non-empty");
    assert!(
        top >= 24,
        "the largest entry is only {top}; the 6-bit alphabet (max 31) is being \
         wasted, so the format's headline precision is not real"
    );
    // And the span across the whole table, not just the peak.
    let lo = SCRUB_JAY_CODEBOOK.iter().flatten().copied().min().unwrap();
    let span = top - lo;
    assert!(
        span >= 24,
        "entries span only {span} of the 63 available levels (lo {lo}, hi {top})"
    );
}

/// The frozen table is what the accessor returns, by reference.
///
/// Not a copy: the table is 256 bytes of read-only data and copying it per call
/// would be pure overhead on a path the GPU will read every block of.
#[test]
fn the_accessor_returns_the_frozen_table_without_copying() {
    let a = scrub_jay_codebook();
    let b = scrub_jay_codebook();
    assert!(
        std::ptr::eq(a, b),
        "the accessor must return the same static table every time"
    );
    assert_eq!(a, &SCRUB_JAY_CODEBOOK);
}

/// The packed form must round trip exactly, including negative entries.
///
/// Negative 6-bit values are the whole reason packing is not a byte cast: -1
/// and 0 differ in their sign bit at position 5, and a naive
/// `as u8 & 0x3F` on a negative i8 would produce 0x3F, not 0x3F-32.
#[test]
fn the_packed_form_round_trips_including_negatives() {
    let packed = pack_codebook();
    // Independent unpacking, written from the 6-bit spec rather than by calling
    // the crate's decoder.
    let mut cursor = 0usize;
    let mut decoded = [[0i8; 16]; 16];
    for book in decoded.iter_mut() {
        for entry in book.iter_mut() {
            // Entries straddle bytes, so pull a 16-bit window and shift.
            let byte = cursor / 8;
            let bit = cursor % 8;
            let lo = *packed.get(byte).unwrap_or(&0) as u16;
            let hi = *packed.get(byte + 1).unwrap_or(&0) as u16;
            let window = lo | (hi << 8);
            let raw = ((window >> bit) & 0x3F) as u8;
            // Sign-extend from 6 bits.
            *entry = if raw & 0x20 != 0 {
                (raw as i8) - 64
            } else {
                raw as i8
            };
            cursor += 6;
        }
    }
    assert_eq!(decoded, SCRUB_JAY_CODEBOOK);
    assert_eq!(cursor, 1536, "packing must consume exactly 1536 bits");
}
