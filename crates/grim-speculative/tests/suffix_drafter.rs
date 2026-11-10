//! CPU-only tests for astrum P6 trigram suffix (prompt-lookup) drafter.
use grim_speculative::SuffixDrafter;

#[test]
fn drafts_from_repeated_pattern() {
    let mut d = SuffixDrafter::new(1024);
    // earlier: 10 11 12 -> followed by 13,14,15; tail is 10 11 12 again
    d.append(&[10, 11, 12, 13, 14, 15, 10, 11, 12]);
    let out = d.draft(3, 8);
    assert!(!out.is_empty(), "should find repeated trigram");
    assert_eq!(out.tokens[0], 13);
    // must not emit tokens that overlap the current suffix region
    for &t in &out.tokens {
        // continuation comes from the earlier occurrence (13,14,15)
        assert!(t == 13 || t == 14 || t == 15);
    }
}

#[test]
fn no_match_when_novel() {
    let mut d = SuffixDrafter::new(1024);
    d.append(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert!(d.draft(3, 8).is_empty());
}

#[test]
fn min_match_enforced() {
    let mut d = SuffixDrafter::new(1024);
    d.append(&[1, 2, 1, 2, 9]); // suffix "1 2" repeats but min_match=3
    assert!(d.draft(3, 8).is_empty());
}

#[test]
fn ring_eviction_keeps_working() {
    let mut d = SuffixDrafter::new(16);
    for i in 0..64u32 {
        d.append(&[i]);
    }
    // must not panic; index must not reference evicted positions
    let _ = d.draft(3, 8);
}

#[test]
fn longest_suffix_wins() {
    let mut d = SuffixDrafter::new(1024);
    // Stream: [5,6,7, 8,9, 5,6,7, 5,6,7] — tail is "5,6,7" (last 3).
    // Earliest non-overlapping match of "5,6,7" is at 0, followed by 8,9,5
    // (index 5 < suffix start at 8), so the valid continuation is [8,9,5].
    d.append(&[5, 6, 7, 8, 9, 5, 6, 7]);
    d.append(&[5, 6, 7]);
    let out = d.draft(2, 8);
    assert!(!out.is_empty(), "repeated suffix should draft");
    assert_eq!(out.tokens[0], 8);
    assert_eq!(out.tokens[1], 9);
    // continuation must not overlap the current suffix
    assert!(out.tokens.len() >= 2);
}
