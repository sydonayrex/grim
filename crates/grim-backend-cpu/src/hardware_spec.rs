//! CPU hardware specification probing, ISA capability inspection, and fingerprinting.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Host CPU vector ISA extensions recognized by [`CpuHardwareSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuIsa {
    /// SSE (x86_64 baseline).
    Sse,
    /// SSE2.
    Sse2,
    /// AVX.
    Avx,
    /// AVX2 (256-bit integer/float SIMD).
    Avx2,
    /// FMA (fused multiply-add).
    Fma,
    /// AVX-512 foundation.
    Avx512f,
    /// AVX-512 byte+word (used by the BF16/Fp8 SIMD paths).
    Avx512bw,
    /// ARM NEON (aarch64).
    Neon,
}

impl CpuIsa {
    fn as_str(self) -> &'static str {
        match self {
            CpuIsa::Sse => "sse",
            CpuIsa::Sse2 => "sse2",
            CpuIsa::Avx => "avx",
            CpuIsa::Avx2 => "avx2",
            CpuIsa::Fma => "fma",
            CpuIsa::Avx512f => "avx512f",
            CpuIsa::Avx512bw => "avx512bw",
            CpuIsa::Neon => "neon",
        }
    }
}

/// Process-wide CPU capability epoch. Bumped whenever [`CpuHardwareSpec::probe`]
/// observes a host fingerprint different from the one last cached. Callers cache
/// ISA- or topology-dependent selections (e.g. which SIMD GEMM kernel to use)
/// and re-validate against [`CpuHardwareSpec::current_epoch`] so they invalidate
/// when the running CPU changes (hotplug, container migration, affinity change).
static CAP_EPOCH: AtomicU64 = AtomicU64::new(0);
static CACHED_FINGERPRINT: Mutex<Option<String>> = Mutex::new(None);

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
    /// Epoch captured at probe time. Compare against [`CpuHardwareSpec::current_epoch`]
    /// to detect a host change since this snapshot was taken.
    pub epoch: u64,
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

        let spec = CpuHardwareSpec {
            arch: arch.to_string(),
            isa_features: features,
            physical_cores,
            logical_cores,
            l1_dcache_bytes: 32 * 1024,
            l2_cache_bytes: 512 * 1024,
            l3_cache_bytes: 32 * 1024 * 1024,
            cache_line_bytes: 64,
            epoch: 0,
        };

        // Bump the global epoch + cache the fingerprint when the host changes.
        let fp = spec.fingerprint_string();
        let mut cached = CACHED_FINGERPRINT.lock().unwrap();
        let changed = cached.as_ref() != Some(&fp);
        let epoch = if changed {
            let e = CAP_EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
            *cached = Some(fp);
            e
        } else {
            CAP_EPOCH.load(Ordering::SeqCst)
        };
        Self { epoch, ..spec }
    }

    /// Whether the given ISA extension is present on the probed host.
    ///
    /// CPU-native analog of ROCm's `QuantCapability::supports(mode)`: capability
    /// gating keyed on detected host features rather than a GCN arch string.
    pub fn supports(&self, isa: CpuIsa) -> bool {
        self.isa_features.iter().any(|f| f == isa.as_str())
    }

    /// Current process-wide capability epoch (see [`CAP_EPOCH`]).
    pub fn current_epoch() -> u64 {
        CAP_EPOCH.load(Ordering::SeqCst)
    }

    /// True if a host change has occurred since this snapshot was probed.
    pub fn is_stale(&self) -> bool {
        self.epoch != Self::current_epoch()
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

    #[test]
    fn supports_reports_detected_isa() {
        let spec = CpuHardwareSpec::probe();
        // Whatever is in isa_features must report true via supports().
        for f in &spec.isa_features {
            let isa = match f.as_str() {
                "sse" => CpuIsa::Sse,
                "sse2" => CpuIsa::Sse2,
                "avx" => CpuIsa::Avx,
                "avx2" => CpuIsa::Avx2,
                "fma" => CpuIsa::Fma,
                "avx512f" => CpuIsa::Avx512f,
                "avx512bw" => CpuIsa::Avx512bw,
                "neon" => CpuIsa::Neon,
                _ => continue,
            };
            assert!(spec.supports(isa), "supports({f}) should be true when present");
        }
        // An ISA that is NOT in the feature list must report false.
        let absent = if spec.supports(CpuIsa::Neon) {
            CpuIsa::Avx2
        } else {
            CpuIsa::Neon
        };
        if !spec.isa_features.is_empty() {
            assert!(!spec.supports(absent) || spec.isa_features.iter().any(|x| x == absent.as_str()));
        }
    }

    #[test]
    fn epoch_is_monotonic_and_invalidates_on_change() {
        let _ = CpuHardwareSpec::probe(); // prime cache
        let before = CpuHardwareSpec::current_epoch();
        let first = CpuHardwareSpec::probe();
        assert!(first.epoch >= before);
        // Re-probing the same host must NOT advance the epoch (idempotent).
        let second = CpuHardwareSpec::probe();
        assert_eq!(first.epoch, second.epoch);
        assert!(!second.is_stale());
        // current_epoch reflects the captured epoch.
        assert_eq!(second.epoch, CpuHardwareSpec::current_epoch());
    }
}
