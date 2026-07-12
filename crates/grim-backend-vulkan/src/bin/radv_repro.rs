//! Minimal standalone RADV compute dispatch repro — pure Rust, no C.
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::{c_void, CStr, CString};
use std::ptr;

// ---------- Pre-compiled SPIR-V for add kernel ----------
// Produced by: glslangValidator -V -o add.spv add.comp
// GLSL: c[i] = a[i] + b[i] for i in 0..4
static SPIRV_ADD: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000b, 0x00000033, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000019,
    0x43667542, 0x00000000, 0x00040006, 0x00000019, 0x00000000, 0x00000063, 0x00030005, 0x0000001b,
    0x00000000, 0x00040005, 0x00000020, 0x41667542, 0x00000000, 0x00040006, 0x00000020, 0x00000000,
    0x00000061, 0x00030005, 0x00000022, 0x00000000, 0x00040005, 0x00000028, 0x42667542, 0x00000000,
    0x00040006, 0x00000028, 0x00000000, 0x00000062, 0x00030005, 0x0000002a, 0x00000000, 0x00040047,
    0x0000000b, 0x0000000b, 0x0000001c, 0x00040047, 0x00000018, 0x00000006, 0x00000004, 0x00030047,
    0x00000019, 0x00000003, 0x00050048, 0x00000019, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x0000001b, 0x00000021, 0x00000002, 0x00040047, 0x0000001b, 0x00000022, 0x00000000, 0x00040047,
    0x0000001f, 0x00000006, 0x00000004, 0x00030047, 0x00000020, 0x00000003, 0x00050048, 0x00000020,
    0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000022, 0x00000021, 0x00000000, 0x00040047,
    0x00000022, 0x00000022, 0x00000000, 0x00040047, 0x00000027, 0x00000006, 0x00000004, 0x00030047,
    0x00000028, 0x00000003, 0x00050048, 0x00000028, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x0000002a, 0x00000021, 0x00000001, 0x00040047, 0x0000002a, 0x00000022, 0x00000000, 0x00040047,
    0x00000032, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006,
    0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009,
    0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000,
    0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0004002b, 0x00000006, 0x00000011, 0x00000004,
    0x00020014, 0x00000012, 0x00030016, 0x00000017, 0x00000020, 0x0003001d, 0x00000018, 0x00000017,
    0x0003001e, 0x00000019, 0x00000018, 0x00040020, 0x0000001a, 0x00000002, 0x00000019, 0x0004003b,
    0x0000001a, 0x0000001b, 0x00000002, 0x00040015, 0x0000001c, 0x00000020, 0x00000001, 0x0004002b,
    0x0000001c, 0x0000001d, 0x00000000, 0x0003001d, 0x0000001f, 0x00000017, 0x0003001e, 0x00000020,
    0x0000001f, 0x00040020, 0x00000021, 0x00000002, 0x00000020, 0x0004003b, 0x00000021, 0x00000022,
    0x00000002, 0x00040020, 0x00000024, 0x00000002, 0x00000017, 0x0003001d, 0x00000027, 0x00000017,
    0x0003001e, 0x00000028, 0x00000027, 0x00040020, 0x00000029, 0x00000002, 0x00000028, 0x0004003b,
    0x00000029, 0x0000002a, 0x00000002, 0x0004002b, 0x00000006, 0x00000030, 0x00000040, 0x0004002b,
    0x00000006, 0x00000031, 0x00000001, 0x0006002c, 0x00000009, 0x00000032, 0x00000030, 0x00000031,
    0x00000031, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b,
    0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f,
    0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x000500ae, 0x00000012, 0x00000013, 0x00000010,
    0x00000011, 0x000300f7, 0x00000015, 0x00000000, 0x000400fa, 0x00000013, 0x00000014, 0x00000015,
    0x000200f8, 0x00000014, 0x000100fd, 0x000200f8, 0x00000015, 0x0004003d, 0x00000006, 0x0000001e,
    0x00000008, 0x0004003d, 0x00000006, 0x00000023, 0x00000008, 0x00060041, 0x00000024, 0x00000025,
    0x00000022, 0x0000001d, 0x00000023, 0x0004003d, 0x00000017, 0x00000026, 0x00000025, 0x0004003d,
    0x00000006, 0x0000002b, 0x00000008, 0x00060041, 0x00000024, 0x0000002c, 0x0000002a, 0x0000001d,
    0x0000002b, 0x0004003d, 0x00000017, 0x0000002d, 0x0000002c, 0x00050081, 0x00000017, 0x0000002e,
    0x00000026, 0x0000002d, 0x00060041, 0x00000024, 0x0000002f, 0x0000001b, 0x0000001d, 0x0000001e,
    0x0003003e, 0x0000002f, 0x0000002e, 0x000100fd, 0x00010038,
];

// ---------- Vulkan FFI declarations ----------

mod vk {
    use std::ffi::c_void;

    pub const VK_SUCCESS: i32 = 0;
    pub const VK_WHOLE_SIZE: u64 = !0u64;

    pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
    pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 5;
    pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 8;
    pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 11;
    pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 12;
    pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 14;
    pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
    pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
    pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
    pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
    pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
    pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 27;
    pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 3;
    pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 2;
    pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;

    pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
    pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0000_0080;
    pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x0000_0002;
    pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x0000_0002;
    pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x0000_0004;
    pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x0000_0001;

    pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0000_0002;
    pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

    pub type VkResult = i32;
    pub type VkFlags = u32;

    #[repr(C)]
    pub struct VkApplicationInfo {
        pub s_type: u32,
        pub p_next: *const c_void,
        pub p_application_name: *const i8,
        pub application_version: u32,
        pub p_engine_name: *const i8,
        pub engine_version: u32,
        pub api_version: u32,
    }

    #[repr(C)]
    pub struct VkInstanceCreateInfo {
        pub s_type: u32,
        pub p_next: *const c_void,
        pub flags: VkFlags,
        pub p_application_info: *const VkApplicationInfo,
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
        pub size: u64,
        pub usage: u32,
        pub sharing_mode: u32,
        pub queue_family_index_count: u32,
        pub p_queue_family_indices: *const u32,
    }

    #[repr(C)]
    pub struct VkMemoryAllocateInfo {
        pub s_type: u32,
        pub p_next: *const c_void,
        pub allocation_size: u64,
        pub memory_type_index: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct VkMemoryRequirements {
        pub size: u64,
        pub alignment: u64,
        pub memory_type_bits: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct VkMemoryType {
        pub property_flags: u32,
        pub heap_index: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct VkMemoryHeap {
        pub size: u64,
        pub flags: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct VkPhysicalDeviceMemoryProperties {
        pub memory_type_count: u32,
        pub memory_types: [VkMemoryType; 32],
        pub memory_heap_count: u32,
        pub memory_heaps: [VkMemoryHeap; 16],
    }

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
        pub offset: u64,
        pub range: u64,
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
        pub p_buffer_info: *const VkDescriptorBufferInfo,
        pub p_image_info: *const c_void,
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
        pub min_image_transfer_granularity_width: u32,
        pub min_image_transfer_granularity_height: u32,
        pub min_image_transfer_granularity_depth: u32,
        pub timestamp_valid_bits: u32,
    }

    // ---------- Function declarations ----------
    #[link(name = "vulkan")]
    unsafe extern "C" {
        pub fn vkCreateInstance(
            p_create_info: *const VkInstanceCreateInfo,
            p_allocator: *const c_void,
            p_instance: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyInstance(instance: u64, p_allocator: *const c_void);
        pub fn vkEnumeratePhysicalDevices(
            instance: u64, p_physical_device_count: *mut u32, p_physical_devices: *mut u64,
        ) -> VkResult;
        pub fn vkGetPhysicalDeviceProperties(physical_device: u64, p_properties: *mut c_void);
        pub fn vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device: u64, p_queue_family_property_count: *mut u32,
            p_queue_family_properties: *mut VkQueueFamilyProperties,
        );
        pub fn vkGetPhysicalDeviceMemoryProperties(
            physical_device: u64, p_memory_properties: *mut VkPhysicalDeviceMemoryProperties,
        );
        pub fn vkCreateDevice(
            physical_device: u64, p_create_info: *const VkDeviceCreateInfo,
            p_allocator: *const c_void, p_device: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyDevice(device: u64, p_allocator: *const c_void);
        pub fn vkGetDeviceQueue(device: u64, queue_family_index: u32, queue_index: u32, p_queue: *mut u64);
        pub fn vkCreateBuffer(
            device: u64, p_create_info: *const VkBufferCreateInfo,
            p_allocator: *const c_void, p_buffer: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyBuffer(device: u64, buffer: u64, p_allocator: *const c_void);
        pub fn vkGetBufferMemoryRequirements(
            device: u64, buffer: u64, p_memory_requirements: *mut VkMemoryRequirements,
        );
        pub fn vkAllocateMemory(
            device: u64, p_allocate_info: *const VkMemoryAllocateInfo,
            p_allocator: *const c_void, p_memory: *mut u64,
        ) -> VkResult;
        pub fn vkFreeMemory(device: u64, memory: u64, p_allocator: *const c_void);
        pub fn vkBindBufferMemory(device: u64, buffer: u64, memory: u64, memory_offset: u64) -> VkResult;
        pub fn vkMapMemory(
            device: u64, memory: u64, offset: u64, size: u64,
            flags: u32, pp_data: *mut *mut c_void,
        ) -> VkResult;
        pub fn vkUnmapMemory(device: u64, memory: u64);
        pub fn vkCreateDescriptorSetLayout(
            device: u64, p_create_info: *const VkDescriptorSetLayoutCreateInfo,
            p_allocator: *const c_void, p_set_layout: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyDescriptorSetLayout(
            device: u64, descriptor_set_layout: u64, p_allocator: *const c_void,
        );
        pub fn vkCreateDescriptorPool(
            device: u64, p_create_info: *const VkDescriptorPoolCreateInfo,
            p_allocator: *const c_void, p_descriptor_pool: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyDescriptorPool(
            device: u64, descriptor_pool: u64, p_allocator: *const c_void,
        );
        pub fn vkAllocateDescriptorSets(
            device: u64, p_allocate_info: *const VkDescriptorSetAllocateInfo,
            p_descriptor_sets: *mut u64,
        ) -> VkResult;
        pub fn vkFreeDescriptorSets(
            device: u64, descriptor_pool: u64,
            descriptor_set_count: u32, p_descriptor_sets: *const u64,
        ) -> VkResult;
        pub fn vkUpdateDescriptorSets(
            device: u64, descriptor_write_count: u32, p_descriptor_writes: *const VkWriteDescriptorSet,
            descriptor_copy_count: u32, p_descriptor_copies: *const c_void,
        );
        pub fn vkCreateShaderModule(
            device: u64, p_create_info: *const VkShaderModuleCreateInfo,
            p_allocator: *const c_void, p_shader_module: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyShaderModule(
            device: u64, shader_module: u64, p_allocator: *const c_void,
        );
        pub fn vkCreatePipelineLayout(
            device: u64, p_create_info: *const VkPipelineLayoutCreateInfo,
            p_allocator: *const c_void, p_pipeline_layout: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyPipelineLayout(
            device: u64, pipeline_layout: u64, p_allocator: *const c_void,
        );
        pub fn vkCreateComputePipelines(
            device: u64, pipeline_cache: u64, create_info_count: u32,
            p_create_infos: *const VkComputePipelineCreateInfo,
            p_allocator: *const c_void, p_pipelines: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyPipeline(device: u64, pipeline: u64, p_allocator: *const c_void);
        pub fn vkCreateCommandPool(
            device: u64, p_create_info: *const VkCommandPoolCreateInfo,
            p_allocator: *const c_void, p_command_pool: *mut u64,
        ) -> VkResult;
        pub fn vkDestroyCommandPool(device: u64, command_pool: u64, p_allocator: *const c_void);
        pub fn vkAllocateCommandBuffers(
            device: u64, p_allocate_info: *const VkCommandBufferAllocateInfo,
            p_command_buffers: *mut u64,
        ) -> VkResult;
        pub fn vkFreeCommandBuffers(
            device: u64, command_pool: u64,
            command_buffer_count: u32, p_command_buffers: *const u64,
        );
        pub fn vkBeginCommandBuffer(
            command_buffer: u64, p_begin_info: *const VkCommandBufferBeginInfo,
        ) -> VkResult;
        pub fn vkEndCommandBuffer(command_buffer: u64) -> VkResult;
        pub fn vkCmdBindPipeline(command_buffer: u64, pipeline_bind_point: u32, pipeline: u64);
        pub fn vkCmdBindDescriptorSets(
            command_buffer: u64, pipeline_bind_point: u32, layout: u64,
            first_set: u32, descriptor_set_count: u32, p_descriptor_sets: *const u64,
            dynamic_offset_count: u32, p_dynamic_offsets: *const u32,
        );
        pub fn vkCmdDispatch(command_buffer: u64, x: u32, y: u32, z: u32);
        pub fn vkQueueSubmit(
            queue: u64, submit_count: u32,
            p_submits: *const VkSubmitInfo, fence: u64,
        ) -> VkResult;
        pub fn vkQueueWaitIdle(queue: u64) -> VkResult;
    }
}

fn main() {
    println!("=== RADV Minimal Compute Repro ===\n");

    // ---------- 1. Instance ----------
    let app_info = vk::VkApplicationInfo {
        s_type: 1,
        p_next: ptr::null(),
        p_application_name: ptr::null(),
        application_version: 0,
        p_engine_name: ptr::null(),
        engine_version: 0,
        api_version: 0x0040_1000, // Vulkan 1.1
    };
    let instance_ci = vk::VkInstanceCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: ptr::null(),
    };
    let mut instance: u64 = 0;
    let res = unsafe { vk::vkCreateInstance(&instance_ci, ptr::null(), &mut instance) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateInstance failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateInstance => {instance:#x}");

    // ---------- 2. Enumerate physical devices ----------
    let mut dev_count: u32 = 0;
    let res = unsafe { vk::vkEnumeratePhysicalDevices(instance, &mut dev_count, ptr::null_mut()) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkEnumeratePhysicalDevices (count) failed: {res}");
        std::process::exit(1);
    }
    if dev_count == 0 {
        eprintln!("No physical devices found.");
        std::process::exit(1);
    }
    println!("[OK] Found {dev_count} physical device(s)");

    let mut phys_devs = vec![0u64; dev_count as usize];
    let res = unsafe { vk::vkEnumeratePhysicalDevices(instance, &mut dev_count, phys_devs.as_mut_ptr()) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkEnumeratePhysicalDevices (list) failed: {res}");
        std::process::exit(1);
    }

    let mut physical_device: u64 = 0;
    let mut chosen_name = String::new();
    for &pd in &phys_devs {
        let mut props = [0u8; 1024];
        unsafe { vk::vkGetPhysicalDeviceProperties(pd, props.as_mut_ptr() as *mut c_void) };
        // deviceName is at offset 20 in VkPhysicalDeviceProperties
        let name = unsafe { CStr::from_ptr(props.as_ptr().add(20) as *const i8) }
            .to_string_lossy()
            .into_owned();
        let is_radv = name.contains("RADV") || name.contains("AMD") || name.contains("Radeon") || name.contains("LLVM");
        println!("  device: {name} (preferred={is_radv})");
        if is_radv && physical_device == 0 {
            physical_device = pd;
            chosen_name = name;
        }
    }
    if physical_device == 0 {
        physical_device = phys_devs[0];
        let mut props = [0u8; 1024];
        unsafe { vk::vkGetPhysicalDeviceProperties(physical_device, props.as_mut_ptr() as *mut c_void) };
        chosen_name = unsafe { CStr::from_ptr(props.as_ptr().add(20) as *const i8) }.to_string_lossy().into_owned();
    }
    println!("[OK] Selected: {chosen_name} ({physical_device:#x})");

    // ---------- 3. Find compute queue family ----------
    let mut qfam_count: u32 = 0;
    unsafe { vk::vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut qfam_count, ptr::null_mut()) };
    let mut qfam_props = vec![vk::VkQueueFamilyProperties {
        queue_flags: 0, queue_count: 0,
        min_image_transfer_granularity_width: 0,
        min_image_transfer_granularity_height: 0,
        min_image_transfer_granularity_depth: 0,
        timestamp_valid_bits: 0,
    }; qfam_count as usize];
    unsafe { vk::vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut qfam_count, qfam_props.as_mut_ptr()) };

    let compute_qfam = qfam_props
        .iter()
        .position(|q| (q.queue_flags & vk::VK_QUEUE_COMPUTE_BIT) != 0)
        .expect("No compute queue family found");
    println!("[OK] Compute queue family: {compute_qfam} (queue_flags={:#x})", qfam_props[compute_qfam].queue_flags);

    // ---------- 4. Create device ----------
    let queue_prio = [1.0f32];
    let queue_ci = vk::VkDeviceQueueCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        queue_family_index: compute_qfam as u32,
        queue_count: 1,
        p_queue_priorities: queue_prio.as_ptr(),
    };
    let device_ci = vk::VkDeviceCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_ci,
        enabled_layer_count: 0,
        pp_enabled_layer_names: ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: ptr::null(),
        p_enabled_features: ptr::null(),
    };
    let mut device: u64 = 0;
    let res = unsafe { vk::vkCreateDevice(physical_device, &device_ci, ptr::null(), &mut device) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateDevice failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateDevice => {device:#x}");

    // ---------- 5. Get queue ----------
    let mut queue: u64 = 0;
    unsafe { vk::vkGetDeviceQueue(device, compute_qfam as u32, 0, &mut queue) };
    println!("[OK] Compute queue: {queue:#x}");

    // ---------- 6. Query memory properties ----------
    let mut mem_props = vk::VkPhysicalDeviceMemoryProperties {
        memory_type_count: 0,
        memory_types: [vk::VkMemoryType { property_flags: 0, heap_index: 0 }; 32],
        memory_heap_count: 0,
        memory_heaps: [vk::VkMemoryHeap { size: 0, flags: 0 }; 16],
    };
    unsafe { vk::vkGetPhysicalDeviceMemoryProperties(physical_device, &mut mem_props) };

    fn find_memory_type(
        mem_props: &vk::VkPhysicalDeviceMemoryProperties,
        type_bits: u32,
        required: u32,
        preferred: u32,
    ) -> Option<u32> {
        let mut best = None;
        let mut best_score = u32::MAX;
        for i in 0..mem_props.memory_type_count {
            let t = &mem_props.memory_types[i as usize];
            if (type_bits & (1 << i)) == 0 {
                continue;
            }
            let score = if t.property_flags & preferred == preferred {
                0u32
            } else if t.property_flags & required == required {
                1
            } else {
                continue;
            };
            if score < best_score {
                best_score = score;
                best = Some(i);
            }
        }
        best
    }

    // ---------- 7. Create 3 buffers (a, b, c) ----------
    let buffer_size: u64 = 4 * 4; // 4 f32 floats

    fn create_buffer(device: u64, size: u64, usage: u32) -> (u64, vk::VkMemoryRequirements) {
        let ci = vk::VkBufferCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: vk::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut buf: u64 = 0;
        let res = unsafe { vk::vkCreateBuffer(device, &ci, ptr::null(), &mut buf) };
        assert_eq!(res, vk::VK_SUCCESS, "vkCreateBuffer failed: {res}");
        let mut reqs = vk::VkMemoryRequirements { size: 0, alignment: 0, memory_type_bits: 0 };
        unsafe { vk::vkGetBufferMemoryRequirements(device, buf, &mut reqs) };
        (buf, reqs)
    }

    let (buf_a, reqs_a) = create_buffer(device, buffer_size, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    let (buf_b, reqs_b) = create_buffer(device, buffer_size, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    let (buf_c, reqs_c) = create_buffer(device, buffer_size, vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    println!("[OK] Created 3 buffers: a={buf_a:#x} b={buf_b:#x} c={buf_c:#x}");

    // ---------- 8. Allocate + bind memory ----------
    fn alloc_and_bind(
        device: u64, buffer: u64, reqs: &vk::VkMemoryRequirements,
        mem_props: &vk::VkPhysicalDeviceMemoryProperties,
    ) -> u64 {
        let preferred = vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let mem_type = find_memory_type(
            mem_props, reqs.memory_type_bits,
            vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            preferred,
        ).expect("No suitable memory type");
        let alloc_info = vk::VkMemoryAllocateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: reqs.size,
            memory_type_index: mem_type,
        };
        let mut mem: u64 = 0;
        let res = unsafe { vk::vkAllocateMemory(device, &alloc_info, ptr::null(), &mut mem) };
        assert_eq!(res, vk::VK_SUCCESS, "vkAllocateMemory failed: {res}");
        let res = unsafe { vk::vkBindBufferMemory(device, buffer, mem, 0) };
        assert_eq!(res, vk::VK_SUCCESS, "vkBindBufferMemory failed: {res}");
        mem
    }

    let mem_a = alloc_and_bind(device, buf_a, &reqs_a, &mem_props);
    let mem_b = alloc_and_bind(device, buf_b, &reqs_b, &mem_props);
    let mem_c = alloc_and_bind(device, buf_c, &reqs_c, &mem_props);
    println!("[OK] Allocated memory: a={mem_a:#x} b={mem_b:#x} c={mem_c:#x}");

    // ---------- 9. Write input data ----------
    let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];

    fn write_to_buffer(device: u64, memory: u64, data: &[f32]) {
        let mut ptr: *mut c_void = ptr::null_mut();
        let res = unsafe { vk::vkMapMemory(device, memory, 0, vk::VK_WHOLE_SIZE, 0, &mut ptr) };
        assert_eq!(res, vk::VK_SUCCESS, "vkMapMemory failed: {res}");
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr() as *const c_void, ptr, data.len());
            vk::vkUnmapMemory(device, memory);
        }
    }

    write_to_buffer(device, mem_a, &a_data);
    write_to_buffer(device, mem_b, &b_data);
    write_to_buffer(device, mem_c, &[0.0f32; 4]);
    println!("[OK] Wrote input: a={a_data:?}, b={b_data:?}");

    println!("[INFO] Step 10: vkCreateDescriptorSetLayout...");

    // ---------- 10. Create descriptor set layout ----------
    let stage_flags = vk::VK_SHADER_STAGE_COMPUTE_BIT;
    let bindings = [
        vk::VkDescriptorSetLayoutBinding {
            binding: 0, descriptor_type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 1, stage_flags, p_immutable_samplers: ptr::null(),
        },
        vk::VkDescriptorSetLayoutBinding {
            binding: 1, descriptor_type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 1, stage_flags, p_immutable_samplers: ptr::null(),
        },
        vk::VkDescriptorSetLayoutBinding {
            binding: 2, descriptor_type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 1, stage_flags, p_immutable_samplers: ptr::null(),
        },
    ];
    let ds_layout_ci = vk::VkDescriptorSetLayoutCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        binding_count: 3,
        p_bindings: bindings.as_ptr(),
    };
    let mut ds_layout: u64 = 0;
    let res = unsafe { vk::vkCreateDescriptorSetLayout(device, &ds_layout_ci, ptr::null(), &mut ds_layout) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateDescriptorSetLayout failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateDescriptorSetLayout => {ds_layout:#x}");

    // ---------- 11. Create descriptor pool ----------
    let pool_sizes = [vk::VkDescriptorPoolSize {
        r#type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptor_count: 3,
    }];
    let ds_pool_ci = vk::VkDescriptorPoolCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        max_sets: 1,
        pool_size_count: 1,
        p_pool_sizes: pool_sizes.as_ptr(),
    };
    let mut ds_pool: u64 = 0;
    let res = unsafe { vk::vkCreateDescriptorPool(device, &ds_pool_ci, ptr::null(), &mut ds_pool) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateDescriptorPool failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateDescriptorPool => {ds_pool:#x}");

    // ---------- 12. Allocate descriptor set ----------
    let ds_alloc_info = vk::VkDescriptorSetAllocateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: ptr::null(),
        descriptor_pool: ds_pool,
        descriptor_set_count: 1,
        p_set_layouts: &ds_layout,
    };
    let mut ds: u64 = 0;
    let res = unsafe { vk::vkAllocateDescriptorSets(device, &ds_alloc_info, &mut ds) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkAllocateDescriptorSets failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkAllocateDescriptorSets => {ds:#x}");

    // ---------- 13. Update descriptor set ----------
    let buf_infos = [
        vk::VkDescriptorBufferInfo { buffer: buf_a, offset: 0, range: vk::VK_WHOLE_SIZE },
        vk::VkDescriptorBufferInfo { buffer: buf_b, offset: 0, range: vk::VK_WHOLE_SIZE },
        vk::VkDescriptorBufferInfo { buffer: buf_c, offset: 0, range: vk::VK_WHOLE_SIZE },
    ];
    let writes = [
        vk::VkWriteDescriptorSet {
            s_type: vk::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(), dst_set: ds, dst_binding: 0, dst_array_element: 0,
            descriptor_count: 1, descriptor_type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(), p_buffer_info: &buf_infos[0], p_texel_buffer_view: ptr::null(),
        },
        vk::VkWriteDescriptorSet {
            s_type: vk::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(), dst_set: ds, dst_binding: 1, dst_array_element: 0,
            descriptor_count: 1, descriptor_type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(), p_buffer_info: &buf_infos[1], p_texel_buffer_view: ptr::null(),
        },
        vk::VkWriteDescriptorSet {
            s_type: vk::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(), dst_set: ds, dst_binding: 2, dst_array_element: 0,
            descriptor_count: 1, descriptor_type: vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(), p_buffer_info: &buf_infos[2], p_texel_buffer_view: ptr::null(),
        },
    ];
    println!("[INFO] About to vkUpdateDescriptorSets with ds={ds:#x}");
    unsafe { vk::vkUpdateDescriptorSets(device, 3, writes.as_ptr(), 0, ptr::null()) };
    println!("[OK] vkUpdateDescriptorSets");

    // ---------- 14. Create shader module (from pre-compiled SPIR-V) ----------
    println!("[INFO] Step 14: vkCreateShaderModule...");
    let shader_ci = vk::VkShaderModuleCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        code_size: SPIRV_ADD.len() * 4,
        p_code: SPIRV_ADD.as_ptr(),
    };
    let mut shader_module: u64 = 0;
    let res = unsafe { vk::vkCreateShaderModule(device, &shader_ci, ptr::null(), &mut shader_module) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateShaderModule failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateShaderModule => {shader_module:#x}");

    // ---------- 15. Create pipeline layout ----------
    let pipe_layout_ci = vk::VkPipelineLayoutCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        set_layout_count: 1,
        p_set_layouts: &ds_layout,
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };
    let mut pipe_layout: u64 = 0;
    let res = unsafe { vk::vkCreatePipelineLayout(device, &pipe_layout_ci, ptr::null(), &mut pipe_layout) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreatePipelineLayout failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreatePipelineLayout => {pipe_layout:#x}");

    // ---------- 16. Create compute pipeline ----------
    println!("[INFO] Step 16: vkCreateComputePipelines...");
    let entry_c = CString::new("main").unwrap();
    let stage_ci = vk::VkPipelineShaderStageCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        stage: vk::VK_SHADER_STAGE_COMPUTE_BIT,
        module: shader_module,
        p_name: entry_c.as_ptr(),
        p_specialization_info: ptr::null(),
    };
    let pipe_ci = vk::VkComputePipelineCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        stage: stage_ci,
        layout: pipe_layout,
        base_pipeline_handle: 0,
        base_pipeline_index: 0,
    };
    let mut pipeline: u64 = 0;
    let res = unsafe { vk::vkCreateComputePipelines(device, 0, 1, &pipe_ci, ptr::null(), &mut pipeline) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateComputePipelines failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateComputePipelines => {pipeline:#x}");

    // ---------- 17. Create command pool ----------
    let cmd_pool_ci = vk::VkCommandPoolCreateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: 0,
        queue_family_index: compute_qfam as u32,
    };
    let mut cmd_pool: u64 = 0;
    let res = unsafe { vk::vkCreateCommandPool(device, &cmd_pool_ci, ptr::null(), &mut cmd_pool) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkCreateCommandPool failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkCreateCommandPool => {cmd_pool:#x}");

    // ---------- 18. Allocate command buffer ----------
    let cmd_alloc_info = vk::VkCommandBufferAllocateInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool: cmd_pool,
        level: 0, // PRIMARY
        command_buffer_count: 1,
    };
    let mut cmd_buf: u64 = 0;
    let res = unsafe { vk::vkAllocateCommandBuffers(device, &cmd_alloc_info, &mut cmd_buf) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkAllocateCommandBuffers failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkAllocateCommandBuffers => {cmd_buf:#x}");

    // ---------- 19. Begin command buffer ----------
    let cmd_begin_info = vk::VkCommandBufferBeginInfo {
        s_type: vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: 0,
        p_inheritance_info: ptr::null(),
    };
    let res = unsafe { vk::vkBeginCommandBuffer(cmd_buf, &cmd_begin_info) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkBeginCommandBuffer failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkBeginCommandBuffer");

    // ---------- 20. Bind pipeline ----------
    unsafe { vk::vkCmdBindPipeline(cmd_buf, 0, pipeline) };
    println!("[OK] vkCmdBindPipeline(COMPUTE, {pipeline:#x})");

    // ---------- 21. Bind descriptor sets ----------
    println!("[INFO] Step 21: vkCmdBindDescriptorSets...");
    unsafe { vk::vkCmdBindDescriptorSets(cmd_buf, 0, pipe_layout, 0, 1, &ds, 0, ptr::null()) };
    println!("[OK] vkCmdBindDescriptorSets({ds:#x}) — NO CRASH");

    // ---------- 22. Dispatch ----------
    unsafe { vk::vkCmdDispatch(cmd_buf, 1, 1, 1) };
    println!("[OK] vkCmdDispatch(1, 1, 1)");

    // ---------- 23. End command buffer ----------
    let res = unsafe { vk::vkEndCommandBuffer(cmd_buf) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkEndCommandBuffer failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkEndCommandBuffer");

    // ---------- 24. Submit ----------
    let submit = vk::VkSubmitInfo {
        s_type: vk::VK_STRUCTURE_TYPE_SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &cmd_buf,
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    };
    let res = unsafe { vk::vkQueueSubmit(queue, 1, &submit, 0) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkQueueSubmit failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkQueueSubmit");

    // ---------- 25. Wait idle ----------
    let res = unsafe { vk::vkQueueWaitIdle(queue) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkQueueWaitIdle failed: {res}");
        std::process::exit(1);
    }
    println!("[OK] vkQueueWaitIdle");

    // ---------- 26. Read back output ----------
    let mut ptr: *mut c_void = ptr::null_mut();
    let res = unsafe { vk::vkMapMemory(device, mem_c, 0, 16, 0, &mut ptr) };
    if res != vk::VK_SUCCESS {
        eprintln!("vkMapMemory (read) failed: {res}");
        std::process::exit(1);
    }
    let mut output = [0.0f32; 4];
    unsafe { ptr::copy_nonoverlapping(ptr as *const c_void, output.as_mut_ptr() as *mut c_void, 4) };
    unsafe { vk::vkUnmapMemory(device, mem_c) };

    println!("\n=== Results ===");
    println!("a     = {a_data:?}");
    println!("b     = {b_data:?}");
    println!("c     = {output:?}");
    println!("exp   = {:?}", [a_data[0]+b_data[0], a_data[1]+b_data[1], a_data[2]+b_data[2], a_data[3]+b_data[3]]);

    let mut ok = true;
    for i in 0..4 {
        let expected = a_data[i] + b_data[i];
        if (output[i] - expected).abs() > 1e-5 {
            println!("FAIL [{i}]: expected {expected}, got {}", output[i]);
            ok = false;
        }
    }

    if ok {
        println!("\nALL PASS — RADV compute dispatch works.");
    } else {
        println!("\nMISMATCH — kernel ran but results are wrong.");
    }

    // ---------- 27. Cleanup ----------
    unsafe {
        vk::vkDestroyCommandPool(device, cmd_pool, ptr::null());
        vk::vkDestroyPipeline(device, pipeline, ptr::null());
        vk::vkDestroyPipelineLayout(device, pipe_layout, ptr::null());
        vk::vkDestroyShaderModule(device, shader_module, ptr::null());
        vk::vkDestroyDescriptorPool(device, ds_pool, ptr::null());
        vk::vkDestroyDescriptorSetLayout(device, ds_layout, ptr::null());
        vk::vkFreeMemory(device, mem_c, ptr::null());
        vk::vkFreeMemory(device, mem_b, ptr::null());
        vk::vkFreeMemory(device, mem_a, ptr::null());
        vk::vkDestroyBuffer(device, buf_c, ptr::null());
        vk::vkDestroyBuffer(device, buf_b, ptr::null());
        vk::vkDestroyBuffer(device, buf_a, ptr::null());
        vk::vkDestroyDevice(device, ptr::null());
        vk::vkDestroyInstance(instance, ptr::null());
    }
    println!("[OK] Cleanup complete.");

    std::process::exit(if ok { 0 } else { 1 });
}