use grim_nn::moe_hybrid::{HybridExecutor, PcieBench};

#[test]
fn test_bandwidth_matched_fetch_fraction() {
    let bench = PcieBench::from_values(/* pcie_bw */ 24.0, /* cpu_ram_bw */ 50.0);
    let frac = bench.hybrid_fetch_fraction();
    assert!((frac - 0.48).abs() < 0.01, "frac={}", frac);

    let (gpu_count, cpu_count) = bench.split_experts(16, frac);
    assert_eq!(gpu_count, 8);  // 16 * 0.48 = 7.68 -> rounds to 8
    assert_eq!(cpu_count, 8);  // remaining on CPU
}

#[test]
fn test_hybrid_executor_split_missing_experts() {
    let executor = HybridExecutor::new(PcieBench::from_values(24.0, 50.0));
    let missing: Vec<usize> = (0..16).collect();
    let (gpu_experts, cpu_experts) = executor.ensure_experts_hybrid(0, &missing);

    assert_eq!(gpu_experts.len(), 8);
    assert_eq!(cpu_experts.len(), 8);
    assert_eq!(gpu_experts, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(cpu_experts, vec![8, 9, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn test_hybrid_executor_zero_missing() {
    let executor = HybridExecutor::new(PcieBench::from_values(24.0, 50.0));
    let (gpu, cpu) = executor.ensure_experts_hybrid(0, &[]);
    assert!(gpu.is_empty());
    assert!(cpu.is_empty());
}
