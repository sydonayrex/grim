//! TDD unit tests for Q5 (`UseCappedResidentPolicy` in `grim-memory::moe_budget`).

use grim_memory::moe_budget::UseCappedResidentPolicy;

#[test]
fn least_used_expert_is_the_unique_victim() {
    let mut p = UseCappedResidentPolicy::new(8, 256);
    // Every expert except 2 gets >= 3 looks, so 2 is the unique minimum.
    for _ in 0..3 {
        p.count_look(0);
        p.count_look(1);
        p.count_look(3);
        p.count_look(4);
        p.count_look(5);
        p.count_look(6);
        p.count_look(7);
    }
    p.count_look(2);
    assert_eq!(p.victim(&[]), Some(2));
}

#[test]
fn halving_strictly_reduces_counts_below_cap() {
    let mut p = UseCappedResidentPolicy::new(8, 256);
    for _ in 0..8 {
        p.count_look(0);
    }
    assert_eq!(p.look(0), 8);
    for _ in 0..256 {
        p.begin_token();
    }
    assert_eq!(p.look(0), 4, "halving must be exact: 8 -> 4");
    assert!(p.look(0) < 8);
}

#[test]
fn mru_tie_break_prefers_most_recently_used_among_equals() {
    let mut p = UseCappedResidentPolicy::new(8, 256);
    // Give every expert other than 5 and 6 >= 2 looks so 5 and 6 tie at minimum count 1.
    for e in [0usize, 1, 2, 3, 4, 7] {
        p.count_look(e);
        p.count_look(e);
    }
    p.count_look(5);
    p.count_look(6); // equal counts (1)
    p.touch(6); // 6 used most recently
    assert_eq!(
        p.victim(&[]),
        Some(5),
        "MRU among equals is kept, LRU among equals is the victim"
    );

    let ordered = p.keep_hottest(&[5, 6, 0]);
    assert_eq!(ordered[0], 0, "highest count expert must rank first");
    assert_eq!(ordered[1], 6, "MRU among equals must rank before LRU");
    assert_eq!(ordered[2], 5);

    p.forget_use();
    assert_eq!(p.look(0), 0);
}

#[test]
fn cap_is_15() {
    let mut p = UseCappedResidentPolicy::new(8, 256);
    for _ in 0..1000 {
        p.count_look(3);
    }
    assert_eq!(p.look(3), 15);
}
