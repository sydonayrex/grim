//! CPU NUMA topology discovery and core affinity mapping.

/// NUMA node and inter-socket topology snapshot for CPU memory placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuNumaTopology {
    /// Number of detected NUMA nodes.
    pub numa_nodes: usize,
    /// Number of logical cores per NUMA node.
    pub cores_per_node: usize,
    /// Symmetric NUMA distance matrix `[node_i][node_j]`.
    pub distance_matrix: Vec<Vec<u32>>,
}

impl CpuNumaTopology {
    /// Probe system NUMA topology (via `/sys/devices/system/node` or system fallback).
    pub fn probe() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Fallback: 1 NUMA node if single-socket or sysfs un-probed
        let mut numa_nodes = 1;
        let mut cores_per_node = logical_cores;

        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            let node_count = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                .count();
            if node_count > 0 {
                numa_nodes = node_count;
                cores_per_node = (logical_cores / numa_nodes).max(1);
            }
        }

        let mut distance_matrix = vec![vec![10u32; numa_nodes]; numa_nodes];
        for i in 0..numa_nodes {
            for j in 0..numa_nodes {
                if i != j {
                    distance_matrix[i][j] = 21; // Standard remote NUMA distance
                }
            }
        }

        CpuNumaTopology {
            numa_nodes,
            cores_per_node,
            distance_matrix,
        }
    }

    /// Map a logical core ID to its corresponding NUMA node index.
    pub fn node_for_core(&self, core_id: usize) -> usize {
        if self.cores_per_node == 0 {
            0
        } else {
            (core_id / self.cores_per_node) % self.numa_nodes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_topology() {
        let topo = CpuNumaTopology::probe();
        assert!(topo.numa_nodes >= 1);
        assert!(topo.cores_per_node >= 1);
        assert_eq!(topo.node_for_core(0), 0);
    }
}
