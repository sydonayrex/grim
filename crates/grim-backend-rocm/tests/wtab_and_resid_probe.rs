use grim_backend_rocm::kernels::comm_fuse::{
    COMM_FUSE_KERNEL_SOURCE, WTAB_PATCH_CAP, WtabPatch, compute_dbg_resid_stat,
};

#[test]
fn comm_fuse_contains_wtab_patch_kernel_and_by_value_struct() {
    assert!(
        COMM_FUSE_KERNEL_SOURCE.contains("grim_wtab_patch"),
        "missing grim_wtab_patch in COMM_FUSE_KERNEL_SOURCE"
    );
    let mut patch = WtabPatch::default();
    let mut fake_table = [0u64; 8];
    let slot_addr = (&mut fake_table[3] as *mut u64) as u64;
    assert!(patch.push(slot_addr, 0xCAFE_BABE_1234_5678));
    assert_eq!(patch.n, 1);
    patch.apply_host_sim();
    assert_eq!(fake_table[3], 0xCAFE_BABE_1234_5678);

    // Filling up to WTAB_PATCH_CAP (24) succeeds; the 25th push returns false so caller flushes.
    for i in 1..WTAB_PATCH_CAP {
        assert!(patch.push(slot_addr, i as u64));
    }
    assert!(!patch.push(slot_addr, 999));
}

/// Ported from Radiance `arch/common/rad_arch.h:55-81`:
/// Transformer hidden states carry 1-2 "massive activation" channels ~100x larger than the bulk.
/// `rms` is dominated by the single spike, whereas `median_abs` isolates the bulk distribution
/// so a broken block delta gain or wrong residual wire is immediately visible.
#[test]
fn radiance_oracle_dbg_resid_stat_separates_massive_activation_from_bulk_median() {
    let mut row = vec![0.10f32; 1024];
    for (i, v) in row.iter_mut().enumerate() {
        if i % 2 == 1 {
            *v = -0.10;
        }
    }
    // Inject a single massive activation channel at index 42
    row[42] = 100.0;

    let stat = compute_dbg_resid_stat(&row);
    assert!((stat.max_abs - 100.0).abs() < 1e-6);
    // Median of |x| is still 0.10 despite the 100.0 outlier spike, while RMS is ~3.126!
    assert!(
        (stat.median_abs - 0.10).abs() < 1e-6,
        "median_abs should reflect bulk 0.10, got {}",
        stat.median_abs
    );
    assert!(
        stat.rms > 3.0,
        "rms should be dominated by the 100.0 spike, got {}",
        stat.rms
    );
}
