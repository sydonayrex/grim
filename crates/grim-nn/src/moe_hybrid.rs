//! Bandwidth-adaptive CPU-GPU hybrid MoE execution ($q^\star$ from FreeToken).
//!
//! When active MoE experts exceed device VRAM, work is partitioned by the
//! bandwidth ratio $q^\star = \text{BW}_{\text{pcie}} / \text{BW}_{\text{cpu\_ram}}$.
//! The first $q^\star$ fraction is fetched to GPU over PCIe, while the remaining
//! overflow experts are computed concurrently on CPU host RAM and merged,
//! perfectly overlapping PCIe transfer latency with CPU compute time.

/// Bandwidth benchmark parameters for calculating optimal fetch fraction $q^\star$.
#[derive(Debug, Clone, Copy)]
pub struct PcieBench {
    /// Measured or configured PCIe bandwidth in GB/s.
    pub pcie_bw_gb_s: f64,
    /// Measured or configured CPU system memory bandwidth in GB/s.
    pub cpu_ram_bw_gb_s: f64,
}

impl PcieBench {
    /// Create benchmark profile from explicit bandwidth values.
    pub fn from_values(pcie_bw_gb_s: f64, cpu_ram_bw_gb_s: f64) -> Self {
        Self {
            pcie_bw_gb_s: pcie_bw_gb_s.max(0.1),
            cpu_ram_bw_gb_s: cpu_ram_bw_gb_s.max(0.1),
        }
    }

    /// Calculate optimal hybrid fetch fraction $q^\star = \text{BW}_{\text{pcie}} / \text{BW}_{\text{cpu}}$.
    pub fn hybrid_fetch_fraction(&self) -> f64 {
        (self.pcie_bw_gb_s / self.cpu_ram_bw_gb_s).clamp(0.0, 1.0)
    }

    /// Split a set of missing expert counts into (GPU fetch count, CPU compute count).
    pub fn split_experts(&self, total_misses: usize, fraction: f64) -> (usize, usize) {
        let gpu = ((total_misses as f64) * fraction).round() as usize;
        let gpu = gpu.min(total_misses);
        let cpu = total_misses.saturating_sub(gpu);
        (gpu, cpu)
    }
}

/// Hybrid CPU-GPU MoE execution coordinator.
pub struct HybridExecutor {
    /// Bandwidth benchmark calibration.
    pub bench: PcieBench,
}

impl HybridExecutor {
    /// Create a new hybrid executor.
    pub fn new(bench: PcieBench) -> Self {
        Self { bench }
    }

    /// Partition missing experts for `layer_id` across GPU fetch and CPU compute sets.
    pub fn ensure_experts_hybrid(
        &self,
        _layer_id: usize,
        missing: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        if missing.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let frac = self.bench.hybrid_fetch_fraction();
        let (gpu_count, _) = self.bench.split_experts(missing.len(), frac);

        let gpu_experts = missing[..gpu_count].to_vec();
        let cpu_experts = missing[gpu_count..].to_vec();

        (gpu_experts, cpu_experts)
    }
}
