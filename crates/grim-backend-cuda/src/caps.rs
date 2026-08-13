//! `CudaCaps`: CUDA hardware capability probe, resource limits, capability gating, and cache key fingerprinting.

use grim_tensor::dtype::QuantFormat;

/// Hardware capabilities and resource ceilings for a CUDA device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaCaps {
    pub device_name: String,
    pub ordinal: usize,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub multi_processor_count: u32,
    pub total_global_mem: u64,
    pub shared_mem_per_block: u32,
    pub max_threads_per_block: u32,
    pub max_grid_dims: [u32; 3],
}

impl CudaCaps {
    /// Probes default capabilities for a device ordinal and name.
    pub fn probe_default(ordinal: usize, device_name: String, major: u32, minor: u32) -> Self {
        Self {
            device_name,
            ordinal,
            compute_major: major,
            compute_minor: minor,
            multi_processor_count: 80,
            total_global_mem: 24 * 1024 * 1024 * 1024,
            shared_mem_per_block: 49152, // 48KB default per block ceiling
            max_threads_per_block: 1024,
            max_grid_dims: [2147483647, 65535, 65535],
        }
    }

    /// Compute 64-bit SeaHash fingerprint of CUDA device capabilities for cache keying.
    pub fn cache_key_hash(&self) -> u64 {
        let mut key_str = format!(
            "{}:{}:{}.{}:{}:{}:{}",
            self.device_name,
            self.ordinal,
            self.compute_major,
            self.compute_minor,
            self.multi_processor_count,
            self.shared_mem_per_block,
            self.max_threads_per_block
        );
        if self.supports_fp8_native() {
            key_str.push_str(":fp8");
        }
        seahash::hash(key_str.as_bytes())
    }

    /// Returns true if native FP8 matrix cores are supported (Compute Capability >= 8.9 / Ada/Hopper).
    pub fn supports_fp8_native(&self) -> bool {
        self.compute_major > 8 || (self.compute_major == 8 && self.compute_minor >= 9)
    }

    /// Capability gate checking if a quantization format is supported on device.
    pub fn supports_quant_format(&self, format: QuantFormat) -> bool {
        match format {
            QuantFormat::Q8_0 | QuantFormat::Q4K | QuantFormat::Q5K | QuantFormat::Q6K | QuantFormat::Iq4Nl => true,
            QuantFormat::Fp8 | QuantFormat::Fp8Block16 => self.supports_fp8_native(),
            _ => true,
        }
    }

    /// Resource limit validation: check if requested shared memory and block threads exceed device ceilings.
    pub fn validate_resource_limits(&self, requested_shared_mem_bytes: u32, threads_per_block: u32) -> bool {
        requested_shared_mem_bytes <= self.shared_mem_per_block
            && threads_per_block <= self.max_threads_per_block
    }
}
