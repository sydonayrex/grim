//! CPU-only tests for astrum P3 frequency-ranked expert trace/profile.
use grim_memory::moe_budget::{ExpertRouteTrace, ResidentProfileFreq};

#[test]
fn frequency_ranked_resident_selection() {
    let mut t = ExpertRouteTrace::new(2, 8);
    for _ in 0..100 {
        t.record(0, 3);
        t.record(0, 7);
    }
    for _ in 0..5 {
        t.record(0, 1);
    }
    let p = ResidentProfileFreq::from_trace(&t, 2);
    assert!(p.is_resident(0, 3));
    assert!(p.is_resident(0, 7));
    assert!(!p.is_resident(0, 1)); // demoted
}

#[test]
fn route_bonus_rank_monotone_and_bounded() {
    let mut t = ExpertRouteTrace::new(1, 8);
    for _ in 0..100 {
        t.record(0, 3);
    }
    for _ in 0..10 {
        t.record(0, 7);
    }
    let p = ResidentProfileFreq::from_trace(&t, 3);
    // rank-0 (expert 3, highest freq) >= rank-1 (expert 7)
    assert!(p.route_bonus(0, 3, 3, 2.0) >= p.route_bonus(0, 7, 3, 2.0));
    // non-resident gets 0
    assert_eq!(p.route_bonus(0, 1, 3, 2.0), 0.0);
    // lambda 0 disables
    assert_eq!(p.route_bonus(0, 3, 3, 0.0), 0.0);
}