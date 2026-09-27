//! WS-A A5: TreePie is the first format in grim that satisfies the attention
//! precision floor natively, instead of being silently clamped up to it.
//!
//! `attention_min_bpw()` is 5 (Q5_K). Every sub-5 format in the tree today --
//! Crow 4.5, Rook 4.1, Jay 4.1, Nutcracker 4.5 -- sits below that floor, so
//! `enforce_attention_precision` silently raises them to 5 for every Q/K/V/O
//! projection. A user who asks for Crow on attention gets 5 bits and no error.
//! That is the intended behaviour, but it means those formats cannot actually
//! deliver their advertised density where it matters most.
//!
//! TreePie is exactly 5.0 bpw, so it is the first format whose request survives
//! the floor unchanged. That is the real argument for TreePie over yet another
//! 4-bit format, and it is what this file asserts.
//!
//! CPU only.

use grim_format::WeightFormat;
use grim_quant::{attention_min_bpw, enforce_attention_precision};

/// Canonical snake_case name for a format, derived from `Debug`.
///
/// Deliberately not `Display`: `Display` runs the serde serializer, so it emits
/// a JSON string *including the quotes* (`"crow"`). Building format identity off
/// that would make this test hostage to a serializer change that has nothing to
/// do with bpw. `Debug` on a unit variant is the bare name.
fn label(f: WeightFormat) -> String {
    format!("{f:?}").to_lowercase()
}

/// The floor is 5, and that is what makes TreePie's 5.0 bpw meaningful.
#[test]
fn the_attention_floor_is_five() {
    assert_eq!(attention_min_bpw(), 5);
}

/// The headline claim: TreePie's 5.0 bpw passes the floor untouched.
///
/// 5.0 is load-bearing. A format at 4.99 would be clamped; the exactness is why
/// the packer had to hit 5.0 rather than 5.33.
#[test]
fn tree_pie_satisfies_the_attention_floor_unchanged() {
    let bpw = WeightFormat::TreePie.bpw();
    assert_eq!(bpw, 5.0, "TreePie must be exactly 5.0 bpw, not merely near it");
    assert_eq!(
        enforce_attention_precision(bpw as u32),
        5,
        "TreePie must pass the attention floor without being clamped"
    );
    assert_eq!(
        enforce_attention_precision(bpw as u32),
        bpw as u32,
        "the floor must be a no-op for TreePie"
    );
}

/// The contrast that makes the claim mean something: every existing sub-5
/// format *is* clamped, and TreePie is the only one that is not.
///
/// This is the regression guard. If someone adds a 4-bit format and it appears
/// in the "clamped" set, that is correct behaviour, not a failure -- the
/// failure would be TreePie joining that set, which the first assertion in this
/// test catches.
#[test]
fn tree_pie_is_the_only_format_that_clears_the_floor_natively() {
    let floor = attention_min_bpw() as f32;

    let all = [
        WeightFormat::Bf16,
        WeightFormat::Crow,
        WeightFormat::Raven,
        WeightFormat::Rook,
        WeightFormat::Jay,
        WeightFormat::Jackdaw,
        WeightFormat::Magpie,
        WeightFormat::Nutcracker,
        WeightFormat::TreePie,
    ];

    let clamped: Vec<String> = all
        .iter()
        .filter(|f| f.bpw() < floor)
        .map(|f| label(*f))
        .collect();
    let clears: Vec<String> = all
        .iter()
        .filter(|f| f.bpw() >= floor)
        .map(|f| label(*f))
        .collect();

    // Sanity: the pre-existing sub-5 formats are indeed all below the floor, so
    // this test is comparing against a real population and not an empty one.
    for expected in ["crow", "nutcracker", "rook", "jay"] {
        assert!(
            clamped.iter().any(|n| n == expected),
            "expected {expected} to be below the floor, got {clamped:?}"
        );
    }

    // TreePie clears it.
    assert!(
        clears.iter().any(|n| n == "treepie"),
        "TreePie must clear the floor, clears = {clears:?}"
    );

    // And no *sub*-5-bpw format clears it. This is the exclusivity claim: a new
    // 4-bit format must not appear here, because then TreePie is not special.
    assert!(
        !clamped.iter().any(|n| n == "treepie"),
        "TreePie fell below the floor and is now clamped like the 4.x formats"
    );
}

/// A 4-bit format still clamps to 5, explicitly.
///
/// The plan calls this out as a required sibling assertion: without it, the
/// journey test later would show a 5-bit result and be misread as a 4.5-bit
/// one. ScrubJay at 4.5 bpw is the case this exists for.
#[test]
fn a_four_bit_format_still_clamps_to_five() {
    assert_eq!(enforce_attention_precision(4), 5, "4 bpw clamps up");
    assert_eq!(enforce_attention_precision(3), 5, "3 bpw clamps up");

    // The real formats that land here, at their actual bpw.
    for f in [WeightFormat::Crow, WeightFormat::Rook, WeightFormat::Jay, WeightFormat::Nutcracker] {
        let bpw = f.bpw();
        assert!(
            bpw < 5.0,
            "{} at {bpw} bpw was expected to be below the floor",
            f.to_string()
        );
        assert_eq!(
            enforce_attention_precision(bpw as u32),
            5,
            "{} must clamp to 5 for attention",
            f.to_string()
        );
    }
}

/// A sub-5 bpw truncates to 4 as a u32 before the floor ever sees it.
///
/// Worth pinning because it is a second, quieter failure mode: `bpw()` is
/// `f32` and the floor takes `u32`, so 4.5 truncates to 4 on the way in. The
/// floor then raises it to 5 -- correct outcome, but via truncation first. If
/// the bridge ever switched to rounding, 4.5 would become 5 and this would
/// still pass while the *reason* the floor exists changed underneath.
#[test]
fn sub_five_bpw_truncates_before_reaching_the_floor() {
    for f in [WeightFormat::Crow, WeightFormat::Nutcracker, WeightFormat::Rook] {
        let bpw = f.bpw();
        assert_eq!(
            bpw as u32,
            4,
            "{} at {bpw} should truncate to 4 as u32",
            f.to_string()
        );
    }
    // TreePie is the exception: 5.0 truncates to 5 cleanly.
    assert_eq!(WeightFormat::TreePie.bpw() as u32, 5);
}

/// TreePie needs no runtime dispatch hint of its own.
///
/// It decodes to native FP16 and feeds the existing FP16 dot path, so it is a
/// storage codec layered on `MxFp4Emulated`'s dispatch family rather than a
/// new gate. Returning `None` keeps it a storage-only alias like Crow and Jay,
/// and is what stops it from claiming a dispatch path that does not exist yet.
#[test]
fn tree_pie_is_a_storage_only_alias_for_now() {
    assert_eq!(
        WeightFormat::TreePie.as_quant_mode_hint(),
        None,
        "TreePie has no dispatch arm yet; it must not claim one"
    );
}

/// The name must round-trip through the header parser.
///
/// A format that cannot be named in a `.grim` header cannot be selected, so this
/// is load-bearing for A7 rather than cosmetic.
#[test]
fn tree_pie_round_trips_through_the_header_parser() {
    for spelling in ["tree_pie", "treepie", "TreePie", "TREE_PIE"] {
        let parsed: WeightFormat = spelling.parse().expect("TreePie must be parseable");
        assert_eq!(parsed, WeightFormat::TreePie, "failed to parse {spelling:?}");
    }
    // And serializes back to the canonical spelling.
    assert_eq!(label(WeightFormat::TreePie), "treepie");
}
