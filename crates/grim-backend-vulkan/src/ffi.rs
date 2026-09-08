//! Vulkan FFI types, constants, and extern "C" declarations.

use std::ffi::c_void;

// Vulkan FFI types and constants

pub type VkFlags = u32;
pub type VkDeviceSize = u64;

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub p_application_info: *const c_void,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const i8,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const i8,
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const i8,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const i8,
    pub p_enabled_features: *const c_void,
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

#[repr(C)]
pub struct VkBufferCopy {
    pub src_offset: VkDeviceSize,
    pub dst_offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VkMemoryType {
    pub property_flags: VkFlags,
    pub heap_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: VkFlags,
}

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; 16],
}

pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;

// Physical device types. Rejects software rasterizers (lavapipe/swiftshader).
pub const VK_PHYSICAL_DEVICE_TYPE_OTHER: u32 = 0;
pub const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
pub const VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: u32 = 3;
pub const VK_PHYSICAL_DEVICE_TYPE_CPU: u32 = 4;

#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: [u8; 256],
    // Remaining fields intentionally omitted; we only read device_type.
}

pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;

pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;

pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;
pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;

pub const VK_SUCCESS: i32 = 0;

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    pub p_immutable_samplers: *const c_void,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub r#type: u32,
    pub descriptor_count: u32,
}

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: u64,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub dst_set: u64,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub descriptor_pool: u64,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const u64,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub code_size: usize,
    pub p_code: *const u32,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub set_layout_count: u32,
    pub p_set_layouts: *const u64,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const c_void,
}

#[repr(C)]
pub struct VkPushConstantRange {
    pub stage_flags: u32,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub stage: u32,
    pub module: u64,
    pub p_name: *const i8,
    pub p_specialization_info: *const c_void,
}

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: u64,
    pub base_pipeline_handle: u64,
    pub base_pipeline_index: i32,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_family_index: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub command_pool: u64,
    pub level: u32,
    pub command_buffer_count: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_inheritance_info: *const c_void,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const u64,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const u64,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity_width: u32,
    pub min_image_transfer_granularity_height: u32,
    pub min_image_transfer_granularity_depth: u32,
}

unsafe extern "C" {
    pub fn vkCreateInstance(
        pCreateInfo: *const VkInstanceCreateInfo,
        pAllocator: *const c_void,
        pInstance: *mut *mut c_void,
    ) -> i32;
    pub fn vkDestroyInstance(instance: *mut c_void, pAllocator: *const c_void);
    pub fn vkEnumeratePhysicalDevices(
        instance: *mut c_void,
        pPhysicalDeviceCount: *mut u32,
        pPhysicalDevices: *mut *mut c_void,
    ) -> i32;
    pub fn vkCreateDevice(
        physicalDevice: *mut c_void,
        pCreateInfo: *const VkDeviceCreateInfo,
        pAllocator: *const c_void,
        pDevice: *mut *mut c_void,
    ) -> i32;
    pub fn vkDestroyDevice(device: *mut c_void, pAllocator: *const c_void);
    pub fn vkCreateBuffer(
        device: *mut c_void,
        pCreateInfo: *const VkBufferCreateInfo,
        pAllocator: *const c_void,
        pBuffer: *mut u64,
    ) -> i32;
    pub fn vkDestroyBuffer(device: *mut c_void, buffer: u64, pAllocator: *const c_void);
    pub fn vkGetBufferMemoryRequirements(
        device: *mut c_void,
        buffer: u64,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );
    pub fn vkAllocateMemory(
        device: *mut c_void,
        pAllocateInfo: *const VkMemoryAllocateInfo,
        pAllocator: *const c_void,
        pMemory: *mut u64,
    ) -> i32;
    pub fn vkFreeMemory(device: *mut c_void, memory: u64, pAllocator: *const c_void);
    pub fn vkBindBufferMemory(
        device: *mut c_void,
        buffer: u64,
        memory: u64,
        memoryOffset: VkDeviceSize,
    ) -> i32;
    pub fn vkMapMemory(
        device: *mut c_void,
        memory: u64,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        flags: VkFlags,
        ppData: *mut *mut c_void,
    ) -> i32;
    pub fn vkUnmapMemory(device: *mut c_void, memory: u64);
    pub fn vkGetPhysicalDeviceMemoryProperties(
        physicalDevice: *mut c_void,
        pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
    );
    pub fn vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice: *mut c_void,
        pQueueFamilyPropertyCount: *mut u32,
        pQueueFamilyProperties: *mut VkQueueFamilyProperties,
    );
    pub fn vkGetPhysicalDeviceProperties(
        physicalDevice: *mut c_void,
        pProperties: *mut VkPhysicalDeviceProperties,
    );
    pub fn vkGetDeviceQueue(
        device: *mut c_void,
        queueFamilyIndex: u32,
        queueIndex: u32,
        pQueue: *mut *mut c_void,
    );
    pub fn vkCreateDescriptorSetLayout(
        device: *mut c_void,
        pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
        pAllocator: *const c_void,
        pSetLayout: *mut u64,
    ) -> i32;
    pub fn vkDestroyDescriptorSetLayout(
        device: *mut c_void,
        descriptorSetLayout: u64,
        pAllocator: *const c_void,
    );
    pub fn vkCreateDescriptorPool(
        device: *mut c_void,
        pCreateInfo: *const VkDescriptorPoolCreateInfo,
        pAllocator: *const c_void,
        pDescriptorPool: *mut u64,
    ) -> i32;
    pub fn vkDestroyDescriptorPool(
        device: *mut c_void,
        descriptorPool: u64,
        pAllocator: *const c_void,
    );
    pub fn vkAllocateDescriptorSets(
        device: *mut c_void,
        pAllocateInfo: *const VkDescriptorSetAllocateInfo,
        pDescriptorSets: *mut u64,
    ) -> i32;
    pub fn vkUpdateDescriptorSets(
        device: *mut c_void,
        descriptorWriteCount: u32,
        pDescriptorWrites: *const VkWriteDescriptorSet,
        descriptorCopyCount: u32,
        pDescriptorCopies: *const c_void,
    );
    pub fn vkCreateShaderModule(
        device: *mut c_void,
        pCreateInfo: *const VkShaderModuleCreateInfo,
        pAllocator: *const c_void,
        pShaderModule: *mut u64,
    ) -> i32;
    pub fn vkDestroyShaderModule(device: *mut c_void, shaderModule: u64, pAllocator: *const c_void);
    pub fn vkCreatePipelineLayout(
        device: *mut c_void,
        pCreateInfo: *const VkPipelineLayoutCreateInfo,
        pAllocator: *const c_void,
        pPipelineLayout: *mut u64,
    ) -> i32;
    pub fn vkDestroyPipelineLayout(
        device: *mut c_void,
        pipelineLayout: u64,
        pAllocator: *const c_void,
    );
    pub fn vkCreateComputePipelines(
        device: *mut c_void,
        pipelineCache: u64,
        createInfoCount: u32,
        pCreateInfos: *const VkComputePipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut u64,
    ) -> i32;
    pub fn vkDestroyPipeline(device: *mut c_void, pipeline: u64, pAllocator: *const c_void);
    pub fn vkCreateCommandPool(
        device: *mut c_void,
        pCreateInfo: *const VkCommandPoolCreateInfo,
        pAllocator: *const c_void,
        pCommandPool: *mut u64,
    ) -> i32;
    pub fn vkDestroyCommandPool(device: *mut c_void, commandPool: u64, pAllocator: *const c_void);
    pub fn vkAllocateCommandBuffers(
        device: *mut c_void,
        pAllocateInfo: *const VkCommandBufferAllocateInfo,
        pCommandBuffers: *mut *mut c_void,
    ) -> i32;
    pub fn vkBeginCommandBuffer(
        commandBuffer: *mut c_void,
        pBeginInfo: *const VkCommandBufferBeginInfo,
    ) -> i32;
    pub fn vkEndCommandBuffer(commandBuffer: *mut c_void) -> i32;
    pub fn vkCmdBindPipeline(commandBuffer: *mut c_void, pipelineBindPoint: u32, pipeline: u64);
    pub fn vkCmdBindDescriptorSets(
        commandBuffer: *mut c_void,
        pipelineBindPoint: u32,
        layout: u64,
        firstSet: u32,
        descriptorSetCount: u32,
        pDescriptorSets: *const u64,
        dynamicOffsetCount: u32,
        pDynamicOffsets: *const u32,
    );
    pub fn vkCmdDispatch(
        commandBuffer: *mut c_void,
        groupCountX: u32,
        groupCountY: u32,
        groupCountZ: u32,
    );
    pub fn vkCmdCopyBuffer(
        commandBuffer: *mut c_void,
        srcBuffer: u64,
        dstBuffer: u64,
        regionCount: u32,
        pRegions: *const VkBufferCopy,
    );
    pub fn vkCmdPushConstants(
        commandBuffer: *mut c_void,
        layout: u64,
        stageFlags: u32,
        offset: u32,
        size: u32,
        pValues: *const c_void,
    );
    pub fn vkQueueSubmit(
        queue: *mut c_void,
        submitCount: u32,
        pSubmits: *const VkSubmitInfo,
        fence: u64,
    ) -> i32;
    pub fn vkQueueWaitIdle(queue: *mut c_void) -> i32;
}
