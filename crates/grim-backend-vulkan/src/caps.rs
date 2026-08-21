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
    /// True when the device supports `buffer_atomic_add_f32` / `OpAtomicFAdd` on SSBOs.
    /// Required by the MoE fused dispatch kernel. AMD RADV only assembles this
    /// instruction on RDNA 3+ (gfx1100+, device_id >= 0x7440); earlier hardware
    /// (including the Raphael Mendocino iGPU at 0x164e) will SIGABRT in ACO.
    pub supports_fp32_atomic_add: bool,
    /// True when device supports `VK_KHR_shader_subgroup_arithmetic` (subgroupAdd, subgroupMax).
    pub supports_subgroup_arithmetic: bool,
    /// Subgroup/Wavefront size (e.g. 32 on RDNA/NVIDIA, 64 on GCN/CDNA/RDNA-wave64).
    pub subgroup_size: u32,
    /// True when device supports `VK_KHR_timeline_semaphore` for pipelined asynchronous queues.
    pub supports_timeline_semaphores: bool,
    /// True when device supports `VK_EXT_external_memory_host` for zero-copy mmap tensor loading.
    pub supports_external_memory_host: bool,
    /// True when device supports `VK_KHR_cooperative_matrix` for hardware matrix cores.
    pub supports_cooperative_matrix: bool,
}

impl VulkanCaps {
    /// Probes default capabilities for a device name and properties.
    pub fn probe_default(
        device_name: String,
        vendor_id: u32,
        device_id: u32,
        driver_version: u32,
    ) -> Self {
        // AMD RDNA 3+ (gfx1100+): device_id in 0x7440–0x75ff range.
        // These support native FP32 atomic add on SSBOs via ACO.
        let supports_fp32_atomic_add =
            vendor_id == 0x1002 && device_id >= 0x7440 && device_id <= 0x75ff;
        let subgroup_size = if vendor_id == 0x1002 {
            32 // AMD RDNA default wave32
        } else if vendor_id == 0x10de {
            32 // NVIDIA warp32
        } else if vendor_id == 0x8086 {
            16 // Intel EU thread / SIMD16
        } else {
            32
        };
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
            supports_fp32_atomic_add,
            supports_subgroup_arithmetic: true,
            subgroup_size,
            supports_timeline_semaphores: true,
            supports_external_memory_host: true,
            supports_cooperative_matrix: vendor_id == 0x10de || vendor_id == 0x1002,
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
