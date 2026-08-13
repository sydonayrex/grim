//! Hardware specification representation and system query snapshot for JIT compiler optimization.

use std::hash::{Hash, Hasher};

use crate::device::capability_profiler::CapabilityProfiler;
use crate::device::probe;
use crate::device::roc_device::RocmDevice;
use crate::peer_access::{LinkType, P2PTopology};

/// Hardware capability snapshot capturing device architecture parameters for JIT specialization.
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// GCN target architecture string (e.g., "gfx1036", "gfx1030").
    pub gcn_arch: String,
    /// Native execution wavefront size in threads (32 for RDNA, 64 for CDNA).
    pub wavefront_size: u32,
    /// Maximum Local Data Share (LDS) shared memory capacity per block in bytes.
    pub max_shared_mem_per_block: u32,
    /// Maximum allowable threads per block.
    pub max_threads_per_block: u32,
    /// Active Compute Unit (CU) count.
    pub cu_count: u32,
    /// Multiprocessor count (identical to `cu_count` on AMD hardware).
    pub multiprocessor_count: u32,
    /// Estimated memory bandwidth in GB/s.
    pub mem_bandwidth_gb_s: f64,
    /// Inter-device P2P topology link matrix.
    pub p2p_topology: P2PTopology,
}

impl PartialEq for HardwareSpec {
    fn eq(&self, other: &Self) -> bool {
        self.gcn_arch == other.gcn_arch
            && self.wavefront_size == other.wavefront_size
            && self.max_shared_mem_per_block == other.max_shared_mem_per_block
            && self.max_threads_per_block == other.max_threads_per_block
            && self.cu_count == other.cu_count
            && self.multiprocessor_count == other.multiprocessor_count
    }
}

impl Eq for HardwareSpec {}

impl Hash for HardwareSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gcn_arch.hash(state);
        self.wavefront_size.hash(state);
        self.max_shared_mem_per_block.hash(state);
        self.max_threads_per_block.hash(state);
        self.cu_count.hash(state);
        self.multiprocessor_count.hash(state);
    }
}

impl From<&RocmDevice> for HardwareSpec {
    fn from(device: &RocmDevice) -> Self {
        let ordinal = device.ordinal();
        let arch = device.gcn_arch().to_string();
        let wf = probe::wavefront_size(ordinal);
        let lds = probe::max_shared_mem(ordinal);
        let max_threads = probe::max_threads_per_block(ordinal);
        let cus = probe::active_cu_count(ordinal);

        let bandwidth = CapabilityProfiler::new()
            .capabilities()
            .get(ordinal)
            .map(|cap| cap.hbm_bandwidth_gbps as f64)
            .unwrap_or(500.0);

        HardwareSpec {
            gcn_arch: arch,
            wavefront_size: wf,
            max_shared_mem_per_block: lds,
            max_threads_per_block: max_threads,
            cu_count: cus,
            multiprocessor_count: cus,
            mem_bandwidth_gb_s: bandwidth,
            p2p_topology: P2PTopology {
                device_count: 1,
                links: vec![vec![LinkType::NoLink]],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_spec_equality_and_hashing() {
        use std::collections::hash_map::DefaultHasher;

        let spec_a = HardwareSpec {
            gcn_arch: "gfx1036".to_string(),
            wavefront_size: 32,
            max_shared_mem_per_block: 393216,
            max_threads_per_block: 1024,
            cu_count: 64,
            multiprocessor_count: 64,
            mem_bandwidth_gb_s: 500.0,
            p2p_topology: P2PTopology {
                device_count: 1,
                links: vec![vec![LinkType::PeerDirect]],
            },
        };

        let spec_b = spec_a.clone();
        assert_eq!(spec_a, spec_b);

        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();
        spec_a.hash(&mut hasher_a);
        spec_b.hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }
}

