//! Cache key primitives for CPU kernel autotuning and specialized routine lookup.

use crate::hardware_spec::CpuHardwareSpec;

/// Cache key uniquely identifying a CPU kernel primitive dispatch by entry name,
/// target ISA, hardware fingerprint, and source specification hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CpuCacheKey {
    /// Primitive or kernel entry point name.
    pub entry: String,
    /// Target ISA string (e.g., "x86_64-avx2").
    pub isa_target: String,
    /// Hardware feature fingerprint string.
    pub hardware_fingerprint: String,
    /// Hash of the primitive specification or kernel source.
    pub source_hash: u64,
}

impl CpuCacheKey {
    /// Construct a `CpuCacheKey` snapshot from a primitive entry name, hardware spec, and source hash.
    pub fn from_spec(entry: &str, spec: &CpuHardwareSpec, source_hash: u64) -> Self {
        let isa = if !spec.isa_features.is_empty() {
            format!("{}-{}", spec.arch, spec.isa_features.join("+"))
        } else {
            spec.arch.clone()
        };

        CpuCacheKey {
            entry: entry.to_string(),
            isa_target: isa,
            hardware_fingerprint: spec.fingerprint_string(),
            source_hash,
        }
    }

    /// Format the cache key into a unique string prefix.
    pub fn to_key_string(&self) -> String {
        format!(
            "grim_cpu_{}_{}_{}_{:016x}",
            self.entry, self.isa_target, self.hardware_fingerprint, self.source_hash
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_from_spec() {
        let spec = CpuHardwareSpec::probe();
        let key = CpuCacheKey::from_spec("quantized_matmul", &spec, 0x123456789abcdef0);
        let s = key.to_key_string();
        assert!(s.contains("quantized_matmul"));
        assert!(s.contains("grim_cpu_"));
    }
}
