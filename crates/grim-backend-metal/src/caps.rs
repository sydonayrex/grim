//! `MetalCaps`: Apple Metal hardware capability probe, resource limits, capability gating, and cache key fingerprinting.

use grim_tensor::dtype::QuantFormat;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

static CAP_EPOCH: AtomicU64 = AtomicU64::new(0);
static CACHED_FINGERPRINT: Mutex<Option<String>> = Mutex::new(None);

/// Hardware capabilities and resource ceilings for an Apple Metal GPU device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalCaps {
    pub device_name: String,
    pub registry_id: u64,
    pub gpu_family: u32,
    pub max_threadgroup_memory_length: u32,
    pub max_threads_per_threadgroup: u32,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,
    pub epoch: u64,
}

impl MetalCaps {
    /// Probes default capabilities for a device name and registry ID.
    pub fn probe_default(registry_id: u64, device_name: String, gpu_family: u32) -> Self {
        let spec = Self {
            device_name,
            registry_id,
            gpu_family,
            max_threadgroup_memory_length: 32768, // 32KB default Metal threadgroup memory ceiling
            max_threads_per_threadgroup: 1024,
            supports_fp16: true,
            supports_bf16: true,
            supports_fp8: gpu_family >= 8, // Apple M3/M4 / GPUFamily8+ supports FP8
            epoch: 0,
        };

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

    /// Current process-wide capability epoch.
    pub fn current_epoch() -> u64 {
        CAP_EPOCH.load(Ordering::SeqCst)
    }

    /// True if device hardware config has changed since this snapshot was probed.
    pub fn is_stale(&self) -> bool {
        self.epoch != Self::current_epoch()
    }

    /// Deterministic fingerprint string for caching and epoch invalidation.
    pub fn fingerprint_string(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.device_name,
            self.registry_id,
            self.gpu_family,
            self.max_threadgroup_memory_length,
            self.max_threads_per_threadgroup
        )
    }

    /// Compute 64-bit SeaHash fingerprint of Metal device capabilities for cache keying.
    pub fn cache_key_hash(&self) -> u64 {
        let mut key_str = self.fingerprint_string();
        if self.supports_fp8 {
            key_str.push_str(":fp8");
        }
        seahash::hash(key_str.as_bytes())
    }

    /// Capability gate checking if a quantization format is supported on device.
    pub fn supports_quant_format(&self, format: QuantFormat) -> bool {
        match format {
            QuantFormat::Q8_0
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K
            | QuantFormat::Iq4Nl => true,
            QuantFormat::Fp8 | QuantFormat::Fp8Block16 => self.supports_fp8,
            _ => true,
        }
    }

    /// Resource limit validation: check if requested threadgroup memory and thread count exceed device ceilings.
    pub fn validate_resource_limits(
        &self,
        requested_threadgroup_mem_bytes: u32,
        threads_per_threadgroup: u32,
    ) -> bool {
        requested_threadgroup_mem_bytes <= self.max_threadgroup_memory_length
            && threads_per_threadgroup <= self.max_threads_per_threadgroup
    }
}
