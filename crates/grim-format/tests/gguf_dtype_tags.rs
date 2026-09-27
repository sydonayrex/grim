//! The GGUF dtype tag table must equal the canonical `GGML_TYPE_*` values.
//!
//! This table was wrong in a way that looked like a corrupt file rather than a
//! bug: codes 16..30 were shifted, so `F64` sat at 20 and `IQ4_NL` at 35. A
//! Qwen3.8-27B whose header stores seven `IQ4_NL` tensors (tag 20) was read as
//! seven F64 tensors, failed grim's dtype/payload byte-length check, and was
//! refused — while llama.cpp loaded the same file without complaint. Only the
//! older download, whose tensors all use tags <= 15 where the table happened to
//! be right, ever worked.
//!
//! Values are transcribed from `ggml/include/ggml.h` (GGML_TYPE_*). If ggml
//! ever renumbers, this test is the thing that should fail first.

use grim_format::gguf::GgufDType;

fn tag(t: &GgufDType) -> u32 {
    GgufDType::from_tag(*t as u32)
        .map(|r| r as u32)
        .unwrap_or(u32::MAX)
}

#[test]
fn gguf_dtype_tags_match_the_canonical_ggml_enum() {
    let canonical: &[(GgufDType, u32)] = &[
        (GgufDType::F32, 0),
        (GgufDType::F16, 1),
        (GgufDType::Q4_0, 2),
        (GgufDType::Q4_1, 3),
        (GgufDType::Q5_0, 6),
        (GgufDType::Q5_1, 7),
        (GgufDType::Q8_0, 8),
        (GgufDType::Q8_1, 9),
        (GgufDType::Q2K, 10),
        (GgufDType::Q3K, 11),
        (GgufDType::Q4K, 12),
        (GgufDType::Q5K, 13),
        (GgufDType::Q6K, 14),
        (GgufDType::Q8K, 15),
        (GgufDType::IQ2_XXS, 16),
        (GgufDType::IQ2_XS, 17),
        (GgufDType::IQ3_XXS, 18),
        (GgufDType::IQ1_S, 19),
        (GgufDType::IQ4_NL, 20),
        (GgufDType::IQ3_S, 21),
        (GgufDType::IQ2_S, 22),
        (GgufDType::IQ4_XS, 23),
        (GgufDType::I8, 24),
        (GgufDType::I16, 25),
        (GgufDType::I32, 26),
        (GgufDType::I64, 27),
        (GgufDType::F64, 28),
        (GgufDType::IQ1_M, 29),
        (GgufDType::BF16, 30),
        (GgufDType::TQ1_0, 34),
        (GgufDType::TQ2_0, 35),
        (GgufDType::MXFP4, 39),
    ];
    for (t, want) in canonical {
        assert_eq!(
            *t as u32, *want,
            "{:?} must be tag {want} (ggml/include/ggml.h GGML_TYPE_*), found {}",
            t, *t as u32
        );
    }
}

/// The two specific inversions that caused the 27B failure, pinned so a
/// partial "fix" cannot reintroduce one of them while fixing the other.
#[test]
fn iq4_nl_and_f64_are_not_transposed() {
    assert_eq!(GgufDType::IQ4_NL as u32, 20, "IQ4_NL is tag 20, not 35");
    assert_eq!(GgufDType::F64 as u32, 28, "F64 is tag 28, not 20");
    // And the decode direction agrees, since a file is read with from_tag.
    assert!(matches!(GgufDType::from_tag(20), Some(GgufDType::IQ4_NL)));
    assert!(matches!(GgufDType::from_tag(28), Some(GgufDType::F64)));
}

/// Types removed upstream must not squat a standard tag — that is precisely
/// how F64 ended up shadowing IQ4_NL. They live on private ids instead.
#[test]
fn removed_types_do_not_squat_standard_tags() {
    for t in [GgufDType::Q4_2, GgufDType::Q8_1Hx] {
        let v = t as u32;
        assert!(
            v > 39,
            "{:?} was removed upstream and must hold a private tag, not {v}",
            t
        );
        // No canonical tag may decode to it.
        for tag in 0u32..=42 {
            if let Some(got) = GgufDType::from_tag(tag) {
                assert_ne!(
                    got, t,
                    "standard tag {tag} decoded to the private/removed type {:?}",
                    t
                );
            }
        }
    }
}

/// Round-trip: every variant must survive tag -> variant -> tag.
#[test]
fn tag_round_trips() {
    for t in [
        GgufDType::F32,
        GgufDType::Q4K,
        GgufDType::Q6K,
        GgufDType::IQ4_NL,
        GgufDType::IQ2_XXS,
        GgufDType::TQ2_0,
        GgufDType::F64,
        GgufDType::Q4_2,
        GgufDType::Q8_1Hx,
        GgufDType::MXFP4,
    ] {
        assert_eq!(tag(&t), t as u32, "{:?} did not round-trip", t);
    }
}
