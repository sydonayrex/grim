//! Bandwidth-adaptive MoE decode miss partition policy (FreeToken q* policy).
//!
//! When serving large MoE models on edge/consumer hardware, the full expert pool exceeds VRAM.
//! For decode steps, each token activates top-k experts. When routed experts miss the GPU LRU
//! cache (total $m$ unique misses), FreeToken dynamically divides them between PCIe DMA cache fills
//! and in-place CPU execution.
//!
//! Because PCIe DMA transfer and CPU execution both read from the same host DRAM subsystem,
//! streaming PCIe transfer leaves residual host bandwidth $B_R = \max(B_H - B_P, 0)$.
//! Balancing the execution times of both branches yields the optimal fill count:
//!
//! $$q^* \approx m \cdot \frac{B_P}{B_H}$$
//!
//! Experts in $\mathcal{F}$ ($|\mathcal{F}| = q^*$) are transferred over PCIe into GPU LRU slots,
//! while experts in $\mathcal{C}$ ($|\mathcal{C}| = m - q^*$) execute concurrently on the CPU.

/// Empirical bandwidth profile of the host system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandwidthProfile {
    /// Measured host-to-device PCIe transfer bandwidth in MB/s (B_P).
    pub pcie_bandwidth_mbps: f64,
    /// Measured effective CPU MoE expert processing bandwidth in MB/s (B_H).
    pub host_bandwidth_mbps: f64,
}

impl Default for BandwidthProfile {
    /// Default conservative profile (e.g., PCIe 4.0 x16 ~25 GB/s, Dual-channel DDR5 ~60 GB/s).
    fn default() -> Self {
        Self {
            pcie_bandwidth_mbps: 25_000.0,
            host_bandwidth_mbps: 60_000.0,
        }
    }
}

impl BandwidthProfile {
    /// Create a new bandwidth profile from measured bandwidth values.
    ///
    /// # Contract
    /// Both `pcie_mbps` and `host_mbps` must be positive.
    pub fn new(pcie_mbps: f64, host_mbps: f64) -> Self {
        assert!(pcie_mbps > 0.0, "pcie_mbps must be > 0");
        assert!(host_mbps > 0.0, "host_mbps must be > 0");
        Self {
            pcie_bandwidth_mbps: pcie_mbps,
            host_bandwidth_mbps: host_mbps,
        }
    }

    /// Calculate optimal cache fill count $q^*$ for $m$ missing experts.
    ///
    /// # Contract
    /// Returns an integer in range `[1, m]` when $m > 0$. When $m = 0$, returns 0.
    /// Always retains at least 1 cache fill when misses exist to ensure the GPU cache warms up.
    pub fn compute_q_star(&self, m: usize) -> usize {
        if m == 0 {
            return 0;
        }
        let ratio = (self.pcie_bandwidth_mbps / self.host_bandwidth_mbps).clamp(0.0, 1.0);
        let q_raw = (m as f64 * ratio).round() as usize;
        // Always retain at least 1 fill to keep GPU cache warming, capped at total misses m
        q_raw.clamp(1, m)
    }

    /// Partition missing expert IDs into GPU cache fill set $\mathcal{F}$ and CPU compute set $\mathcal{C}$.
    ///
    /// # Contract
    /// `misses` contains unique missing expert indices.
    /// Returns `(gpu_fill_set, cpu_compute_set)` where `gpu_fill_set.len() == q*`.
    pub fn partition_misses(&self, misses: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let m = misses.len();
        if m == 0 {
            return (Vec::new(), Vec::new());
        }
        let q = self.compute_q_star(m);
        let gpu_fills = misses[..q].to_vec();
        let cpu_computes = misses[q..].to_vec();
        (gpu_fills, cpu_computes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_q_star_ratios() {
        // PCIe 25 GB/s, Host 100 GB/s -> ratio 0.25
        let profile = BandwidthProfile::new(25_000.0, 100_000.0);
        assert_eq!(profile.compute_q_star(0), 0);
        // m = 4 -> 4 * 0.25 = 1
        assert_eq!(profile.compute_q_star(4), 1);
        // m = 8 -> 8 * 0.25 = 2
        assert_eq!(profile.compute_q_star(8), 2);
        // m = 12 -> 12 * 0.25 = 3
        assert_eq!(profile.compute_q_star(12), 3);

        // Even with small ratio, retains at least 1 fill
        let low_pcie = BandwidthProfile::new(10_000.0, 100_000.0);
        assert_eq!(low_pcie.compute_q_star(2), 1);
    }

    #[test]
    fn test_partition_misses() {
        let profile = BandwidthProfile::new(50_000.0, 100_000.0); // 50% split
        let misses = vec![10, 20, 30, 40];
        let (fills, cpu) = profile.partition_misses(&misses);
        assert_eq!(fills, vec![10, 20]);
        assert_eq!(cpu, vec![30, 40]);
    }
}
