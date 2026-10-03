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
//! WhiteRaven (670), Raven (669), and WhiteCrow (660) all have file loaders
//! today. ForestRaven (672) and GreyRaven (671) are refused with a message
//! naming the format; this test pins that they are refused rather than
//! silently decoded.

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
fn the_rest_of_the_series_is_refused_naming_the_format() {
    // Raven (669) and WhiteCrow (660) became loadable once their QuantFormats
    // landed; only these two genuinely have no consumer yet. Both still count
    // as grim-native tags: a stock GGUF reader must reject them the same way.
    for d in [
        GgufDType::ForestRaven,
        GgufDType::GreyRaven,
    ] {
        let dt = map_gguf_dtype_to_storage(d);
        match dt.storage {
            DTypeStorage::Unsupported(u) => {
                let reason = u.reason.to_lowercase();
                assert!(
                    reason.contains(&d.display_name().to_lowercase())
                        || reason.contains("raven")
                        || reason.contains("crow"),
                    "{} must be refused naming itself, got: {reason}",
                    d.display_name()
                );
            },
            other => panic!(
                "{} has no loader and must not resolve to a real storage: {other:?}",
                d.display_name()
            ),
        }
    }
}

#[test]
fn raven_and_whitecrow_resolve_to_loadable_storages() {
    use grim_tensor::dtype::Storage;
    let raven = map_gguf_dtype_to_storage(GgufDType::Raven);
    assert!(matches!(
        raven.storage,
        Storage::FloatPack(grim_tensor::dtype::FloatPackScheme::Fp8)
    ));
    let crow = map_gguf_dtype_to_storage(GgufDType::WhiteCrow);
    assert!(matches!(crow.storage, Storage::W4A4OstQuant(_)));
}
