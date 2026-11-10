//! CPU-only tests for astrum P5 online draft-source policy (E/cost gate).
use grim_speculative::{DraftPolicy, DraftSource};

#[test]
fn initial_selection_is_mtp() {
    let mut p = DraftPolicy::new(0.05);
    assert_eq!(p.select(100), DraftSource::Mtp);
    assert_eq!(p.select(1000), DraftSource::Mtp);
    assert_eq!(p.select(3000), DraftSource::Mtp);
}

#[test]
fn switches_to_clearly_better_source() {
    let mut p = DraftPolicy::new(0.05);
    // bucket 0: MTP mediocre, Suffix much better
    for _ in 0..20 {
        p.observe_in_bucket(DraftSource::Mtp, 0, 10, 10.0); // 1.0 tok/ms
        p.observe_in_bucket(DraftSource::Suffix, 0, 30, 10.0); // 3.0 tok/ms
    }
    assert_eq!(p.select(100), DraftSource::Suffix);
}

#[test]
fn hysteresis_keeps_choice_within_margin() {
    let mut p = DraftPolicy::new(0.10);
    // establish Suffix as winner in bucket 0
    for _ in 0..30 {
        p.observe_in_bucket(DraftSource::Mtp, 0, 10, 10.0); // 1.0
        p.observe_in_bucket(DraftSource::Suffix, 0, 20, 10.0); // 2.0
    }
    assert_eq!(p.select(100), DraftSource::Suffix);
    // move suffix only slightly worse than mtp but within 10% margin -> keep Suffix
    for _ in 0..80 {
        p.observe_in_bucket(DraftSource::Mtp, 0, 105, 100.0); // 1.05
        p.observe_in_bucket(DraftSource::Suffix, 0, 100, 100.0); // 1.0
    }
    assert_eq!(p.select(100), DraftSource::Suffix);
    // now make mtp clearly better (>10% margin over suffix) -> switch to Mtp
    for _ in 0..80 {
        p.observe_in_bucket(DraftSource::Mtp, 0, 150, 100.0); // 1.5
        p.observe_in_bucket(DraftSource::Suffix, 0, 100, 100.0); // 1.0
    }
    assert_eq!(p.select(100), DraftSource::Mtp);
}

#[test]
fn buckets_are_independent() {
    let mut p = DraftPolicy::new(0.05);
    // bucket 0 (ctx <=512): suffix wins
    for _ in 0..30 {
        p.observe_in_bucket(DraftSource::Mtp, 0, 10, 10.0);
        p.observe_in_bucket(DraftSource::Suffix, 0, 40, 10.0);
    }
    // bucket 2 (ctx >=2049): mtp wins
    for _ in 0..30 {
        p.observe_in_bucket(DraftSource::Mtp, 2, 40, 10.0);
        p.observe_in_bucket(DraftSource::Suffix, 2, 10, 10.0);
    }
    assert_eq!(p.select(100), DraftSource::Suffix); // bucket 0
    assert_eq!(p.select(3000), DraftSource::Mtp); // bucket 2
    // bucket 1 unseen -> default Mtp
    assert_eq!(p.select(1000), DraftSource::Mtp);
}

#[test]
fn output_neutral_contract_is_documented_by_api_shape() {
    // DraftPolicy only returns a source choice; it has no token/logit fields.
    let mut p = DraftPolicy::new(0.05);
    let src = p.observe_and_select(DraftSource::Mtp, 64, 2, 1.0);
    assert!(matches!(src, DraftSource::Mtp | DraftSource::Suffix));
}
