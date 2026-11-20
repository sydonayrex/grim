use grim_backend_rocm::device::moe_hybrid_exec::{
    PcieLatencyModel, compute_pcie_share, select_top5_of_6_faster_experts,
};

#[test]
fn oracle_compute_pcie_share_never_idles_cpu_pool() {
    let share = compute_pcie_share(28.0, 25.0);
    assert!((share - (28.0 / 59.25)).abs() < 1e-5, "unexpected share {share}");
    assert!(share >= 0.10 && share <= 0.95);

    let share_low = compute_pcie_share(28.0, 10.0);
    assert!((share_low - (28.0 / 36.0)).abs() < 1e-5);

    let share_high = compute_pcie_share(28.0, 50.0);
    assert!((share_high - (28.0 / 120.5)).abs() < 1e-5);
}

#[test]
fn oracle_latency_model_subset_splitter_minimizes_makespan() {
    let model = PcieLatencyModel {
        pcie_setup_us: 8.0,
        pcie_bw_bytes_per_us: 28_000.0,
        cpu_setup_us: 5.0,
        cpu_bw_bytes_per_us: 25_000.0,
    };
    let sizes = vec![2 * 1024 * 1024; 4];
    let mask = model.optimal_split_mask(&sizes);
    assert_eq!(mask.count_ones(), 2);
}

#[test]
fn oracle_top5_of_6_drops_cold_tail_and_renormalizes() {
    let routed = vec![
        (10, 0.30f32),
        (20, 0.25),
        (30, 0.18),
        (40, 0.12),
        (50, 0.10),
        (60, 0.05),
    ];
    let kept_cold = select_top5_of_6_faster_experts(&routed, |_e| false);
    assert_eq!(kept_cold.len(), 5);
    let sum_cold: f32 = kept_cold.iter().map(|x| x.1).sum();
    assert!((sum_cold - 1.0).abs() < 1e-5);

    let kept_hot = select_top5_of_6_faster_experts(&routed, |e| e == 60);
    assert_eq!(kept_hot.len(), 6);
}
