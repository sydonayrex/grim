//! `VulkanCaps`: Physical device property probe, limits, capability gating, and cache key hashing.

use grim_tensor::dtype::QuantFormat;

/// Hardware capabilities and resource ceilings for a Vulkan physical device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VulkanCaps {
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub max_shared_memory_bytes: u32,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,
}

impl VulkanCaps {
    /// Probes default capabilities for a device name and properties.
    pub fn probe_default(
        device_name: String,
        vendor_id: u32,
        device_id: u32,
        driver_version: u32,
    ) -> Self {
        Self {
            device_name,
            vendor_id,
            device_id,
            driver_version,
            max_shared_memory_bytes: 32768, // 32KB default VkPhysicalDeviceLimits ceiling
            max_workgroup_invocations: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_fp16: true,
            supports_bf16: true,
            supports_fp8: false,
        }
    }

    /// Compute 64-bit SeaHash fingerprint of device capabilities for cache keying.
    pub fn cache_key_hash(&self) -> u64 {
        let mut key_str = format!(
            "{}:{}:{}:{}:{}:{}",
            self.device_name,
            self.vendor_id,
            self.device_id,
            self.driver_version,
            self.max_shared_memory_bytes,
            self.max_workgroup_invocations
        );
        if self.supports_fp16 {
            key_str.push_str(":fp16");
        }
        if self.supports_bf16 {
            key_str.push_str(":bf16");
        }
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

    /// Resource limit validation: check if requested shared memory and workgroup size exceed device ceilings.
    pub fn validate_resource_limits(
        &self,
        requested_shared_mem_bytes: u32,
        workgroup_invocations: u32,
    ) -> bool {
        requested_shared_mem_bytes <= self.max_shared_memory_bytes
            && workgroup_invocations <= self.max_workgroup_invocations
    }
}
