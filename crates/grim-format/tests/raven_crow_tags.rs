//! Grim-native Raven/Crow tags: the space, and the one loader that exists.
//!
//! Two properties are pinned here, both of which are easy to break silently:
//!
//! 1. **The tag space is partitioned by weight width.** 660..668 is 4-bit,
//!    669..677 is 8-bit. If a future entry is added at the wrong base, a reader
//!    that gates on the block (`is_grim_native`) still works, but a reader that
//!    assumes "4-bit iff tag < 669" starts classifying wrongly.
//!
//! 2. **These tags are not GGUF.** They live far outside ggml's assigned range
//!    precisely so no other tool can ever claim them: a stock GGUF reader
//!    rejects them as unknown types, which is what makes a checkpoint carrying
//!    one unloadable outside grim *by construction* rather than by a policy
//!    someone can forget to apply.
//!
//! Every member of the Raven/Crow series now has a file loader. ForestRaven
//! (672) was the last to land: symmetric per-row absmax INT8 in a framed
//! blob, decoded on the host. No member is refused anymore; this test pins
//! that the refusals are gone rather than silently reintroduced.

use grim_format::gguf::{map_gguf_dtype_to_storage, GgufDType};
use grim_tensor::dtype::{FloatPackScheme, Storage as DTypeStorage};

/// Highest tag upstream ggml has ever assigned, as of ROCm 7.x / llama.cpp.
/// 660 sits 400+ above this, which is the whole safety argument.
const MAX_UPSTREAM_GGML_TAG: u32 = 143;

#[test]
fn every_raven_crow_tag_is_far_above_the_upstream_range() {
    for d in [
        GgufDType::WhiteCrow,
        GgufDType::Raven,
        GgufDType::WhiteRaven,
        GgufDType::GreyRaven,
        GgufDType::GreyRavenHw,
        GgufDType::ForestRaven,
    ] {
        assert!(
            d.tag() > MAX_UPSTREAM_GGML_TAG,
            "{} carries tag {} which upstream could assign at any time",
            d.display_name(),
            d.tag()
        );
        assert!(
            d.is_grim_native(),
            "{} must be flagged grim-native",
            d.display_name()
        );
    }
    // And a real GGUF format must NOT be.
    for d in [GgufDType::F32, GgufDType::Q4K, GgufDType::MXFP4, GgufDType::NVFP4] {
        assert!(!d.is_grim_native(), "{} is real GGUF", d.display_name());
    }
}

#[test]
fn the_tag_space_is_partitioned_by_weight_width() {
    // 669 = 660 + 9: nine 4-bit slots are reserved before the 8-bit block.
    assert_eq!(GgufDType::GRIM_NATIVE_8BIT_BASE - GgufDType::GRIM_NATIVE_4BIT_BASE, 9);
    assert_eq!(GgufDType::GRIM_NATIVE_4BIT_BASE, 660);
    assert_eq!(GgufDType::GRIM_NATIVE_8BIT_BASE, 669);

    // The base of each block is the only member of its width.
    assert!(GgufDType::WhiteCrow.is_grim_native());
    for d in [
        GgufDType::Raven,
        GgufDType::WhiteRaven,
        GgufDType::GreyRaven,
        GgufDType::GreyRavenHw,
        GgufDType::ForestRaven,
    ] {
        assert!(
            d.tag() >= GgufDType::GRIM_NATIVE_8BIT_BASE,
            "{} is 8-bit and must sit at or above the 8-bit base",
            d.display_name()
        );
    }

    // Density: WhiteCrow is 4-bit plus per-group overhead, so claiming exactly
    // 4.0 bpw would understate VRAM. The 8-bit family is a flat 8.
    let wc = GgufDType::WhiteCrow.grim_native_bpw().expect("WhiteCrow has a density");
    assert!(wc > 4.0 && wc < 4.3, "WhiteCrow bpw {wc} should carry scale+zero overhead");
    for d in [
        GgufDType::Raven,
        GgufDType::WhiteRaven,
        GgufDType::ForestRaven,
    ] {
        assert_eq!(
            d.grim_native_bpw(),
            Some(8.0),
            "{} is a bare byte per weight",
            d.display_name()
        );
    }
}

#[test]
fn tags_round_trip_through_from_tag() {
    for d in [
        GgufDType::WhiteCrow,
        GgufDType::Raven,
        GgufDType::WhiteRaven,
        GgufDType::GreyRaven,
        GgufDType::GreyRavenHw,
        GgufDType::ForestRaven,
    ] {
        assert_eq!(
            GgufDType::from_tag(d.tag()),
            Some(d),
            "{} lost its tag {} on the way back",
            d.display_name(),
            d.tag()
        );
    }
    // The 4-bit and 8-bit bases must not be interchangeable.
    assert_ne!(GgufDType::WhiteCrow.tag(), GgufDType::Raven.tag());
}

#[test]
fn whiteraven_resolves_to_the_blocked_fp8_storage() {
    // The loader. A blocked tensor must never resolve to plain Fp8: the two
    // hold the same bytes in different orders, so a mixup decodes to finite,
    // plausible, wrong weights rather than faulting.
    let dt = map_gguf_dtype_to_storage(GgufDType::WhiteRaven);
    assert_eq!(dt.arith, grim_tensor::ArithType::U8);
    assert_eq!(
        dt.storage,
        DTypeStorage::FloatPack(FloatPackScheme::Fp8Blocked16),
        "WhiteRaven must load as blocked FP8, not row-major"
    );
}

#[test]
fn every_series_member_resolves_to_a_real_storage() {
    // The series is complete: no member may resolve to Unsupported. A format
    // that regresses to Unsupported loads nowhere, and the failure surfaces
    // as a consumer-side refusal far from this table.
    for d in [
        GgufDType::WhiteCrow,
        GgufDType::Raven,
        GgufDType::WhiteRaven,
        GgufDType::GreyRaven,
        GgufDType::GreyRavenHw,
        GgufDType::ForestRaven,
    ] {
        let dt = map_gguf_dtype_to_storage(d);
        assert!(
            !matches!(dt.storage, DTypeStorage::Unsupported(_)),
            "{} must resolve to a real storage",
            d.display_name()
        );
    }
}

#[test]
fn raven_whitecrow_greyraven_and_forestraven_resolve_to_loadable_storages() {
    use grim_tensor::dtype::Storage;
    let raven = map_gguf_dtype_to_storage(GgufDType::Raven);
    assert!(matches!(
        raven.storage,
        Storage::FloatPack(grim_tensor::dtype::FloatPackScheme::Fp8)
    ));
    let crow = map_gguf_dtype_to_storage(GgufDType::WhiteCrow);
    assert!(matches!(crow.storage, Storage::W4A4OstQuant(_)));
    // GreyRaven decodes on the host; the GPU kernel is still the probe, but
    // the tag resolves to a real storage so the file loads.
    let grey = map_gguf_dtype_to_storage(GgufDType::GreyRaven);
    assert!(matches!(
        grey.storage,
        Storage::Block(grim_tensor::dtype::BlockDtype::Fp8Sparse24)
    ));
    // ForestRaven: per-row absmax INT8 in the framed blob. The tag HAS a
    // meaning (this storage); the GGUF container just cannot express the
    // framing, so GGUF-direct refuses while .grim serves.
    let forest = map_gguf_dtype_to_storage(GgufDType::ForestRaven);
    assert!(matches!(
        forest.storage,
        Storage::Block(grim_tensor::dtype::BlockDtype::Int8PerChannel)
    ));
}
