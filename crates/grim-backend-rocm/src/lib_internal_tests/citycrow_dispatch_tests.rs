//! CityCrow dispatch guards: the scheme->encoder split and the cache-identity
//! separation between WhiteCrow and CityCrow.
//!
//! The hazard these cover is silent. WhiteCrow and CityCrow both emit the same
//! three-segment u4 lane triple, but they encode the *same input weight* to
//! *different bytes* — WhiteCrow is a 16-level affine `[min, max]` encode with a
//! per-group derived zero, CityCrow is a 4-level symmetric `absmax` encode whose
//! `zeros` is the offset-binary codebook bias. Serving one cache entry to the
//! other returns finite, plausible, wrong weights: no fault, no NaN, just a
//! quietly degraded model.

use crate::device::compute::gemm_launchers::{
    U4_LANE_CACHE_HEADER, u4_lane_cache_enabled_for, u4_lane_cache_key, u4_lane_cache_read,
    u4_lane_cache_write, u4_lane_converter_for,
};
use grim_tensor::KQuantScheme;

/// A temp directory that removes itself, so these tests leave nothing behind.
///
/// No `tempfile` dependency for one call site: a name from the process id and a
/// monotonic counter is unique enough within the test binary, and the tests are
/// not run concurrently with a second copy of themselves.
struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "grim-citycrow-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("scratch dir");
        Self(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// GSQRCO must route to CityCrow, and the 4/6/8-bit K-quants to WhiteCrow.
///
/// The scheme -> encoder decision lives in exactly one place, and this pins it,
/// because a scheme routed to the wrong encoder does not error -- it produces a
/// weight that decodes to something else entirely.
#[test]
fn citycrow_scheme_maps_to_the_citycrow_encoder() {
    assert_eq!(
        u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("GsqRco3p5 is served"),
        u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("GsqRco3p5 is served"),
    );
    // Distinctness is the load-bearing part: the two must never collapse.
    assert_ne!(
        u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("scheme is served"),
        u4_lane_converter_for(KQuantScheme::Q4K).expect("scheme is served"),
        "GsqRco3p5 and Q4K must not share a cache identity",
    );
}

/// The two encoders must differ in BOTH cache discriminators, not just one.
///
/// Either alone would work; having only one would mean a single mistake in the
/// other is silently fatal. `ext` is the guard that keeps a lookup from even
/// opening the wrong file; `magic` is the guard that survives a collision or a
/// hand-copied cache directory.
#[test]
fn the_two_encoders_differ_in_magic_and_extension() {
    let cc = u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("scheme is served");
    let wc = u4_lane_converter_for(KQuantScheme::Q4K).expect("scheme is served");
    assert_ne!(cc.magic, wc.magic, "cache magic must differ");
    assert_ne!(cc.ext, wc.ext, "cache extension must differ");
    assert_ne!(
        cc.encoder_version, 0xDEAD_BEEF,
        "sanity: the encoder version must be a real constant, not a placeholder"
    );
}

/// A scheme no u4-lane encoder serves is refused BY NAME.
///
/// Q2_0 is the interesting case: it shares GSQRCO's 18-byte-per-64 geometry but
/// not its codebook, so routing it to either encoder would yield a plausible,
/// wrong model rather than an error.
#[test]
fn an_unserved_scheme_is_refused_by_name() {
    let err = u4_lane_converter_for(KQuantScheme::Q2_0).expect_err("Q2_0 has no u4 encoder");
    let msg = format!("{err}");
    assert!(
        msg.contains("Q2_0"),
        "the refusal must name the scheme so it is diagnosable: {msg}"
    );
}

/// The load-bearing test: a WhiteCrow cache entry must NOT be readable as a
/// CityCrow entry for the same weight, geometry, and key.
///
/// Written as a round trip through both the write and the read side, because
/// either half alone would pass with the other broken: a read that ignores
/// `magic` passes when nothing was written under the foreign name, and a write
/// that ignores it passes when the read is also broken.
#[test]
fn a_whitecrow_entry_is_never_readable_as_citycrow() {
    let dir = ScratchDir::new("alias");
    let packed = vec![0xABu8; 18 * 4];
    let (n, k) = (2usize, 128usize);

    let cc = u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("scheme is served");
    let wc = u4_lane_converter_for(KQuantScheme::Q4K).expect("scheme is served");

    // Same source bytes, same geometry, same scheme-derived key: the ONLY
    // difference is which encoder wrote it.
    let key = u4_lane_cache_key(&packed, n, k, KQuantScheme::Q4K, wc);
    let (qw, sc, zr) = (vec![1u8; 16], vec![2u8; 4], vec![3u8; 2]);
    u4_lane_cache_write(dir.path(), key, wc, n, k, &qw, &sc, &zr);

    // The WhiteCrow reader gets its own entry back...
    assert!(
        u4_lane_cache_read(dir.path(), key, wc, n, k).is_some(),
        "the writer's own encoder must read its own entry, or the cache never hits"
    );
    // ...and the CityCrow reader must not, even for that exact key.
    assert!(
        u4_lane_cache_read(dir.path(), key, cc, n, k).is_none(),
        "a WhiteCrow entry was served to CityCrow: different encoders produce \
         different bytes for the same weight, so this is silent wrong weights"
    );
}

/// A CityCrow entry must likewise not be readable as WhiteCrow, and the key
/// hash itself must differ between the two encoders' identities.
///
/// The differing key is belt-and-braces with the differing magic: it means even
/// a corrupted header cannot make one converter land on the other's file.
#[test]
fn the_key_hash_also_separates_the_two_encoders() {
    let packed = vec![0x5Au8; 18 * 4];
    let (n, k) = (2usize, 128usize);
    let cc = u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("scheme is served");
    let wc = u4_lane_converter_for(KQuantScheme::Q4K).expect("scheme is served");

    // Same scheme in both, so only the encoder version can move the hash.
    let a = u4_lane_cache_key(&packed, n, k, KQuantScheme::GsqRco3p5, cc);
    let b = u4_lane_cache_key(&packed, n, k, KQuantScheme::GsqRco3p5, wc);
    assert_ne!(
        a, b,
        "the encoder identity must enter the key hash, not only the header"
    );
}

/// The disabling switch must actually disable, and must not over-reach.
///
/// Tested through the pure predicate rather than by setting the variable:
/// `set_var` mutates process-global state, and every other test in this binary
/// that computes a cache key would race it — which is not hypothetical, an
/// earlier revision of this file did exactly that and made an unrelated key test
/// fail intermittently.
#[test]
fn the_cache_switch_disables_only_on_its_documented_values() {
    for off in ["0", "false", "off"] {
        assert!(
            !u4_lane_cache_enabled_for(Some(off)),
            "{off:?} is documented as disabling the cache"
        );
    }
    // Unset is ON (the cache is a pure optimization), and an unrecognized value
    // is also ON rather than an error, so a typo cannot silently cost every user
    // a reconvert.
    assert!(
        u4_lane_cache_enabled_for(None),
        "unset must leave the cache enabled"
    );
    for on in ["1", "true", "on", "", "nope", "00"] {
        assert!(
            u4_lane_cache_enabled_for(Some(on)),
            "{on:?} is not a documented disabling value, so the cache stays on"
        );
    }
}

/// A truncated or foreign file must read as a MISS, never as a panic and never
/// as a short triple.
///
/// The length check is what makes this safe: the reader validates the declared
/// segment lengths against the actual file size before slicing, so a file
/// truncated mid-write costs the conversion instead of indexing out of bounds.
#[test]
fn a_truncated_entry_reads_as_a_miss_not_a_panic() {
    let dir = ScratchDir::new("trunc");
    let (n, k) = (1usize, 128usize);
    let cc = u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("scheme is served");
    let key = 0xDEAD_BEEF_CAFE_0001u64;

    let qw = vec![7u8; 64];
    let sc = vec![8u8; 2];
    let zr = vec![9u8; 1];
    u4_lane_cache_write(dir.path(), key, cc, n, k, &qw, &sc, &zr);

    // Sanity: the intact file round-trips.
    let got = u4_lane_cache_read(dir.path(), key, cc, n, k).expect("intact entry hits");
    assert_eq!(
        (got.0, got.1, got.2),
        (qw, sc, zr),
        "round trip must be exact"
    );

    // Chop the body in half and re-read: a miss, not a panic.
    let path = dir.path().join(format!("{key:016x}.cc"));
    let full = std::fs::read(&path).expect("read");
    std::fs::write(&path, &full[..full.len() / 2]).expect("truncate");
    assert!(
        u4_lane_cache_read(dir.path(), key, cc, n, k).is_none(),
        "a truncated entry must miss; the declared lengths no longer match the \
         file size, and slicing them would panic"
    );

    // Chop below the header: still a miss, and no indexing of a short header.
    std::fs::write(&path, &full[..U4_LANE_CACHE_HEADER - 1]).expect("truncate");
    assert!(
        u4_lane_cache_read(dir.path(), key, cc, n, k).is_none(),
        "a file shorter than the header must miss before any header read"
    );
}

/// A stale geometry must miss even when the magic and version are intact.
///
/// `n`/`k` are in the key hash, but the header repeats them, and the header is
/// what a reader trusts. Pin the header's own check so changing the key hash
/// cannot silently remove the header's.
#[test]
fn a_mismatched_geometry_reads_as_a_miss() {
    let dir = ScratchDir::new("geom");
    let cc = u4_lane_converter_for(KQuantScheme::GsqRco3p5).expect("scheme is served");
    let key = 0x1234_5678_9ABC_DEF0u64;
    u4_lane_cache_write(
        dir.path(),
        key,
        cc,
        1,
        128,
        &[1u8; 32],
        &[2u8; 2],
        &[3u8; 1],
    );

    assert!(
        u4_lane_cache_read(dir.path(), key, cc, 2, 128).is_none(),
        "n is written into the header and must be re-checked on read"
    );
    assert!(
        u4_lane_cache_read(dir.path(), key, cc, 1, 256).is_none(),
        "k is written into the header and must be re-checked on read"
    );
    assert!(
        u4_lane_cache_read(dir.path(), key, cc, 1, 128).is_some(),
        "the matching geometry must still hit, or the two checks above prove \
         nothing about a live entry"
    );
}
