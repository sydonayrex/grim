use grim_backend_rocm::memory::hugepage::{
    compute_cgroup_v2_reclaimable_bytes, compute_host_pin_budget,
    should_madvise_hugepage_from_proc, should_skip_uma_host_pin,
};

#[test]
fn oracle_buddyinfo_and_thp_guard_prevents_compaction_freeze() {
    let buddy_plenty =
        "Node 0, zone   Normal      1      2      3      4      5      6      7      8      9    100     50\n";
    let buddy_fragmented =
        "Node 0, zone   Normal     99     88     77     66     55     44     33     22     11      2      0\n";

    assert!(!should_madvise_hugepage_from_proc(
        "[always] madvise never",
        buddy_plenty,
        64 * 1024 * 1024
    ));
    assert!(should_madvise_hugepage_from_proc(
        "always [madvise] never",
        buddy_plenty,
        64 * 1024 * 1024
    ));
    assert!(!should_madvise_hugepage_from_proc(
        "always [madvise] never",
        buddy_fragmented,
        64 * 1024 * 1024
    ));
}

#[test]
fn oracle_uma_unpinned_and_soft_hard_pin_budget() {
    assert!(should_skip_uma_host_pin("gfx1103", true));
    assert!(should_skip_uma_host_pin("gfx1151", true));
    assert!(!should_skip_uma_host_pin("gfx1201", false));

    let gib = 1024 * 1024 * 1024u64;
    let discrete = compute_host_pin_budget(64 * gib, 50 * gib, None, false);
    assert_eq!(discrete.soft_reserve_bytes, (64 * gib) / 5);
    assert_eq!(discrete.hard_reserve_bytes, (64 * gib) / 10);
    assert_eq!(discrete.soft_budget_bytes, 50 * gib - (64 * gib) / 5);
    assert_eq!(discrete.hard_budget_bytes, 50 * gib - (64 * gib) / 10);

    let uma = compute_host_pin_budget(128 * gib, 110 * gib, None, true);
    assert_eq!(uma.uma_reserve_bytes, (128 * gib) / 12);
}

#[test]
fn oracle_cgroup_v2_reclaimable_subtracts_dirty_and_writeback() {
    let stat = "\
anon 104857600
file 524288000
inactive_file 400000000
file_dirty 50000000
file_writeback 25000000
";
    assert_eq!(compute_cgroup_v2_reclaimable_bytes(stat), 325_000_000);
}
