//! CPU hardware specification probing, ISA capability inspection, and fingerprinting.

use std::hash::{Hash, Hasher};

/// CPU hardware capability snapshot capturing ISA extensions, core count, and cache hierarchy.
#[derive(Debug, Clone)]
pub struct CpuHardwareSpec {
    /// Host CPU architecture string ("x86_64", "aarch64", "unknown").
    pub arch: String,
    /// Detected ISA vector extensions (e.g., ["avx2", "avx512f", "fma"]).
    pub isa_features: Vec<String>,
    /// Number of physical CPU cores.
    pub physical_cores: usize,
    /// Number of logical threads (`std::thread::available_parallelism`).
    pub logical_cores: usize,
    /// L1 data cache size per core in bytes (default 32 KiB if un-probed).
    pub l1_dcache_bytes: usize,
    /// L2 cache size per core in bytes (default 512 KiB if un-probed).
    pub l2_cache_bytes: usize,
    /// L3 shared cache size in bytes (default 32 MiB if un-probed).
    pub l3_cache_bytes: usize,
    /// Cache line size in bytes (default 64).
    pub cache_line_bytes: usize,
}

impl CpuHardwareSpec {
    /// Probe the current host CPU for active vector ISA extensions and topology.
    pub fn probe() -> Self {
        let arch = if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "unknown"
        };

        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse") {
                features.push("sse".to_string());
            }
            if is_x86_feature_detected!("sse2") {
                features.push("sse2".to_string());
            }
            if is_x86_feature_detected!("avx") {
                features.push("avx".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if is_x86_feature_detected!("fma") {
                features.push("fma".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                features.push("avx512f".to_string());
            }
            if is_x86_feature_detected!("avx512bw") {
                features.push("avx512bw".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if cfg!(target_feature = "neon") {
                features.push("neon".to_string());
            }
        }

        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let physical_cores = (logical_cores / 2).max(1);

        CpuHardwareSpec {
            arch: arch.to_string(),
            isa_features: features,
            physical_cores,
            logical_cores,
            l1_dcache_bytes: 32 * 1024,
            l2_cache_bytes: 512 * 1024,
            l3_cache_bytes: 32 * 1024 * 1024,
            cache_line_bytes: 64,
        }
    }

    /// Format a deterministic hardware fingerprint string.
    pub fn fingerprint_string(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.arch,
            self.isa_features.join("+"),
            self.logical_cores,
            self.l3_cache_bytes,
            self.cache_line_bytes
        )
    }
}

impl PartialEq for CpuHardwareSpec {
    fn eq(&self, other: &Self) -> bool {
        self.arch == other.arch
            && self.isa_features == other.isa_features
            && self.logical_cores == other.logical_cores
            && self.l3_cache_bytes == other.l3_cache_bytes
    }
}

impl Eq for CpuHardwareSpec {}

impl Hash for CpuHardwareSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.arch.hash(state);
        self.isa_features.hash(state);
        self.logical_cores.hash(state);
        self.l3_cache_bytes.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_valid_spec() {
        let spec = CpuHardwareSpec::probe();
        assert!(!spec.arch.is_empty());
        assert!(spec.logical_cores >= 1);
        assert!(!spec.fingerprint_string().is_empty());
    }

    #[test]
    fn equality_and_hashing() {
        use std::collections::hash_map::DefaultHasher;

        let spec_a = CpuHardwareSpec::probe();
        let spec_b = spec_a.clone();
        assert_eq!(spec_a, spec_b);

        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();
        spec_a.hash(&mut hasher_a);
        spec_b.hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }
}
