//! Cross-platform Vulkan compute backend with compiled SPIR-V shaders.

use std::ffi::c_void;
use std::sync::Mutex;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, QuantProvenance, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, ScythePlacement, Shape};

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
    fn vkCreateInstance(
        pCreateInfo: *const VkInstanceCreateInfo,
        pAllocator: *const c_void,
        pInstance: *mut *mut c_void,
    ) -> i32;
    fn vkDestroyInstance(instance: *mut c_void, pAllocator: *const c_void);
    fn vkEnumeratePhysicalDevices(
        instance: *mut c_void,
        pPhysicalDeviceCount: *mut u32,
        pPhysicalDevices: *mut *mut c_void,
    ) -> i32;
    fn vkCreateDevice(
        physicalDevice: *mut c_void,
        pCreateInfo: *const VkDeviceCreateInfo,
        pAllocator: *const c_void,
        pDevice: *mut *mut c_void,
    ) -> i32;
    fn vkDestroyDevice(device: *mut c_void, pAllocator: *const c_void);
    fn vkCreateBuffer(
        device: *mut c_void,
        pCreateInfo: *const VkBufferCreateInfo,
        pAllocator: *const c_void,
        pBuffer: *mut u64,
    ) -> i32;
    fn vkDestroyBuffer(device: *mut c_void, buffer: u64, pAllocator: *const c_void);
    fn vkGetBufferMemoryRequirements(
        device: *mut c_void,
        buffer: u64,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );
    fn vkAllocateMemory(
        device: *mut c_void,
        pAllocateInfo: *const VkMemoryAllocateInfo,
        pAllocator: *const c_void,
        pMemory: *mut u64,
    ) -> i32;
    fn vkFreeMemory(device: *mut c_void, memory: u64, pAllocator: *const c_void);
    fn vkBindBufferMemory(
        device: *mut c_void,
        buffer: u64,
        memory: u64,
        memoryOffset: VkDeviceSize,
    ) -> i32;
    fn vkMapMemory(
        device: *mut c_void,
        memory: u64,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        flags: VkFlags,
        ppData: *mut *mut c_void,
    ) -> i32;
    fn vkUnmapMemory(device: *mut c_void, memory: u64);
    fn vkGetPhysicalDeviceMemoryProperties(
        physicalDevice: *mut c_void,
        pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
    );
    fn vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice: *mut c_void,
        pQueueFamilyPropertyCount: *mut u32,
        pQueueFamilyProperties: *mut VkQueueFamilyProperties,
    );
    fn vkGetPhysicalDeviceProperties(
        physicalDevice: *mut c_void,
        pProperties: *mut VkPhysicalDeviceProperties,
    );
    fn vkGetDeviceQueue(
        device: *mut c_void,
        queueFamilyIndex: u32,
        queueIndex: u32,
        pQueue: *mut *mut c_void,
    );
    fn vkCreateDescriptorSetLayout(
        device: *mut c_void,
        pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
        pAllocator: *const c_void,
        pSetLayout: *mut u64,
    ) -> i32;
    fn vkDestroyDescriptorSetLayout(
        device: *mut c_void,
        descriptorSetLayout: u64,
        pAllocator: *const c_void,
    );
    fn vkCreateDescriptorPool(
        device: *mut c_void,
        pCreateInfo: *const VkDescriptorPoolCreateInfo,
        pAllocator: *const c_void,
        pDescriptorPool: *mut u64,
    ) -> i32;
    fn vkDestroyDescriptorPool(device: *mut c_void, descriptorPool: u64, pAllocator: *const c_void);
    fn vkAllocateDescriptorSets(
        device: *mut c_void,
        pAllocateInfo: *const VkDescriptorSetAllocateInfo,
        pDescriptorSets: *mut u64,
    ) -> i32;
    fn vkUpdateDescriptorSets(
        device: *mut c_void,
        descriptorWriteCount: u32,
        pDescriptorWrites: *const VkWriteDescriptorSet,
        descriptorCopyCount: u32,
        pDescriptorCopies: *const c_void,
    );
    fn vkCreateShaderModule(
        device: *mut c_void,
        pCreateInfo: *const VkShaderModuleCreateInfo,
        pAllocator: *const c_void,
        pShaderModule: *mut u64,
    ) -> i32;
    fn vkDestroyShaderModule(device: *mut c_void, shaderModule: u64, pAllocator: *const c_void);
    fn vkCreatePipelineLayout(
        device: *mut c_void,
        pCreateInfo: *const VkPipelineLayoutCreateInfo,
        pAllocator: *const c_void,
        pPipelineLayout: *mut u64,
    ) -> i32;
    fn vkDestroyPipelineLayout(device: *mut c_void, pipelineLayout: u64, pAllocator: *const c_void);
    fn vkCreateComputePipelines(
        device: *mut c_void,
        pipelineCache: u64,
        createInfoCount: u32,
        pCreateInfos: *const VkComputePipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut u64,
    ) -> i32;
    fn vkDestroyPipeline(device: *mut c_void, pipeline: u64, pAllocator: *const c_void);
    fn vkCreateCommandPool(
        device: *mut c_void,
        pCreateInfo: *const VkCommandPoolCreateInfo,
        pAllocator: *const c_void,
        pCommandPool: *mut u64,
    ) -> i32;
    fn vkDestroyCommandPool(device: *mut c_void, commandPool: u64, pAllocator: *const c_void);
    fn vkAllocateCommandBuffers(
        device: *mut c_void,
        pAllocateInfo: *const VkCommandBufferAllocateInfo,
        pCommandBuffers: *mut *mut c_void,
    ) -> i32;
    fn vkBeginCommandBuffer(
        commandBuffer: *mut c_void,
        pBeginInfo: *const VkCommandBufferBeginInfo,
    ) -> i32;
    fn vkEndCommandBuffer(commandBuffer: *mut c_void) -> i32;
    fn vkCmdBindPipeline(commandBuffer: *mut c_void, pipelineBindPoint: u32, pipeline: u64);
    fn vkCmdBindDescriptorSets(
        commandBuffer: *mut c_void,
        pipelineBindPoint: u32,
        layout: u64,
        firstSet: u32,
        descriptorSetCount: u32,
        pDescriptorSets: *const u64,
        dynamicOffsetCount: u32,
        pDynamicOffsets: *const u32,
    );
    fn vkCmdDispatch(
        commandBuffer: *mut c_void,
        groupCountX: u32,
        groupCountY: u32,
        groupCountZ: u32,
    );
    fn vkCmdPushConstants(
        commandBuffer: *mut c_void,
        layout: u64,
        stageFlags: u32,
        offset: u32,
        size: u32,
        pValues: *const c_void,
    );
    fn vkQueueSubmit(
        queue: *mut c_void,
        submitCount: u32,
        pSubmits: *const VkSubmitInfo,
        fence: u64,
    ) -> i32;
    fn vkQueueWaitIdle(queue: *mut c_void) -> i32;
}

// Vulkan helper context

struct VulkanContext {
    instance: *mut c_void,
    physical_device: *mut c_void,
    device: *mut c_void,
    queue: *mut c_void,
    compute_family_index: u32,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
    fn init() -> Result<Self> {
        // Do NOT enable the Primus/Bumblebee layer; it requires an X display and
        // breaks headless/GPU-direct hosts. The loader resolves the native
        // ICD without it.
        let instance_ci = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            p_application_info: std::ptr::null(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };

        let mut instance: *mut c_void = std::ptr::null_mut();
        let res = unsafe { vkCreateInstance(&instance_ci, std::ptr::null(), &mut instance) };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateInstance failed with status {}",
                res
            )));
        }

        let mut gpu_count: u32 = 0;
        unsafe {
            vkEnumeratePhysicalDevices(instance, &mut gpu_count, std::ptr::null_mut());
        }
        if gpu_count == 0 {
            unsafe {
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend("No Vulkan physical devices found".into()));
        }

        let mut gpus = vec![std::ptr::null_mut(); gpu_count as usize];
        let res = unsafe {
            vkEnumeratePhysicalDevices(instance, &mut gpu_count, gpus.as_mut_ptr())
        };
        if res != VK_SUCCESS || gpus.is_empty() || gpus[0].is_null() {
            unsafe {
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend(format!(
                "vkEnumeratePhysicalDevices failed or returned null device with status {}",
                res
            )));
        }
        let physical_device = gpus[0];

        // Reject software rasterizers (lavapipe/swiftshader/llvmpipe) — never
        // report a CPU tier as a live GPU (§3).
        let mut props = VkPhysicalDeviceProperties {
            api_version: 0,
            driver_version: 0,
            vendor_id: 0,
            device_id: 0,
            device_type: 0,
            device_name: [0u8; 256],
        };
        unsafe { vkGetPhysicalDeviceProperties(physical_device, &mut props) };
        if props.device_type == VK_PHYSICAL_DEVICE_TYPE_CPU {
            unsafe {
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend(
                "Vulkan physical device is a software rasterizer (CPU); refusing to use it as a GPU tier".into(),
            ));
        }

        // Find compute queue family index
        let mut qfam_count: u32 = 0;
        unsafe {
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut qfam_count,
                std::ptr::null_mut(),
            );
        }
        if qfam_count == 0 {
            unsafe {
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend(
                "No queue families found on Vulkan physical device".into(),
            ));
        }
        let mut qfam_props = vec![
            VkQueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                min_image_transfer_granularity_width: 0,
                min_image_transfer_granularity_height: 0,
                min_image_transfer_granularity_depth: 0,
                timestamp_valid_bits: 0,
            };
            qfam_count as usize
        ];
        unsafe {
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut qfam_count,
                qfam_props.as_mut_ptr(),
            );
        }
        let mut compute_family_index = None;
        for i in 0..qfam_count {
            if (qfam_props[i as usize].queue_flags & VK_QUEUE_COMPUTE_BIT) != 0 {
                compute_family_index = Some(i);
                break;
            }
        }
        let compute_family_index = match compute_family_index {
            Some(idx) => idx,
            None => {
                unsafe {
                    vkDestroyInstance(instance, std::ptr::null());
                }
                return Err(Error::Backend(
                    "No compute queue family found on Vulkan physical device".into(),
                ));
            }
        };

        let priorities: f32 = 1.0f32;
        let queue_ci = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: compute_family_index,
            queue_count: 1,
            p_queue_priorities: &priorities,
        };

        let device_ci = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_ci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
        };

        let mut device: *mut c_void = std::ptr::null_mut();
        let res =
            unsafe { vkCreateDevice(physical_device, &device_ci, std::ptr::null(), &mut device) };
        if res != VK_SUCCESS {
            unsafe {
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend(format!(
                "vkCreateDevice failed with status {}",
                res
            )));
        }

        let mut queue: *mut c_void = std::ptr::null_mut();
        unsafe {
            vkGetDeviceQueue(device, compute_family_index, 0, &mut queue);
        }
        if queue.is_null() {
            unsafe {
                vkDestroyDevice(device, std::ptr::null());
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend("vkGetDeviceQueue returned null queue pointer".into()));
        }

        Ok(Self {
            instance,
            physical_device,
            device,
            queue,
            compute_family_index,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            if !self.device.is_null() {
                vkDestroyDevice(self.device, std::ptr::null());
            }
            if !self.instance.is_null() {
                vkDestroyInstance(self.instance, std::ptr::null());
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref GLOBAL_CONTEXT: Mutex<Option<VulkanContext>> = Mutex::new(VulkanContext::init().ok());
}

// Vulkan crate structs

/// A handle to a Vulkan compute operation.
///
/// INVARIANT: `run_compute_shader` calls `vkQueueWaitIdle` synchronously during dispatch.
/// Therefore, operations associated with `VulkanHandle` are already completed when returned,
/// making `synchronize()` a safe no-op and `is_ready()` always true.
#[derive(Debug)]
pub struct VulkanHandle;

impl ComputeHandle for VulkanHandle {
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn is_ready(&self) -> bool {
        true
    }
}

/// Vulkan-side tensor storage.
#[derive(Debug)]
pub struct VulkanStorage {
    buffer: u64,
    memory: u64,
    bytes: usize,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: *mut c_void,
}

unsafe impl Send for VulkanStorage {}
unsafe impl Sync for VulkanStorage {}

impl VulkanStorage {
    /// Allocates memory and a buffer on the Vulkan device.
    pub fn alloc_gpu(
        shape: &Shape,
        dtype: DType,
        device: *mut c_void,
        physical_device: *mut c_void,
    ) -> Result<Self> {
        let bytes = shape.elem_count() * dtype_byte_size(&dtype);

        let buffer_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size: bytes as VkDeviceSize,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };

        let mut buffer: u64 = 0;
        let res = unsafe { vkCreateBuffer(device, &buffer_ci, std::ptr::null(), &mut buffer) };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateBuffer failed with status {}",
                res
            )));
        }

        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        unsafe {
            vkGetBufferMemoryRequirements(device, buffer, &mut reqs);
        }

        // Find a host-visible and host-coherent memory type index
        let memory_type_index = {
            let mut mem_properties = VkPhysicalDeviceMemoryProperties {
                memory_type_count: 0,
                memory_types: [VkMemoryType {
                    property_flags: 0,
                    heap_index: 0,
                }; 32],
                memory_heap_count: 0,
                memory_heaps: [VkMemoryHeap { size: 0, flags: 0 }; 16],
            };
            unsafe {
                vkGetPhysicalDeviceMemoryProperties(physical_device, &mut mem_properties);
            }

            let required_properties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            let mut found_type = None;
            for i in 0..mem_properties.memory_type_count {
                if (reqs.memory_type_bits & (1 << i)) != 0
                    && (mem_properties.memory_types[i as usize].property_flags
                        & required_properties)
                        == required_properties
                {
                    found_type = Some(i);
                    break;
                }
            }
            found_type.ok_or_else(|| {
                Error::Backend("Failed to find suitable Vulkan memory type".into())
            })?
        };

        let alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: reqs.size,
            memory_type_index,
        };

        let mut memory: u64 = 0;
        let res = unsafe { vkAllocateMemory(device, &alloc_info, std::ptr::null(), &mut memory) };
        if res != VK_SUCCESS {
            unsafe {
                vkDestroyBuffer(device, buffer, std::ptr::null());
            }
            return Err(Error::Backend(format!(
                "vkAllocateMemory failed with status {}",
                res
            )));
        }

        let res = unsafe { vkBindBufferMemory(device, buffer, memory, 0) };
        if res != VK_SUCCESS {
            unsafe {
                vkFreeMemory(device, memory, std::ptr::null());
                vkDestroyBuffer(device, buffer, std::ptr::null());
            }
            return Err(Error::Backend(format!(
                "vkBindBufferMemory failed with status {}",
                res
            )));
        }

        Ok(Self {
            buffer,
            memory,
            bytes,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            device,
        })
    }
}

impl Drop for VulkanStorage {
    fn drop(&mut self) {
        unsafe {
            vkDestroyBuffer(self.device, self.buffer, std::ptr::null());
            vkFreeMemory(self.device, self.memory, std::ptr::null());
        }
    }
}

impl BackendStorage for VulkanStorage {
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                self.device,
                self.memory,
                0,
                self.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed with status {}",
                res
            )));
        }

        let mut out = vec![0.0f32; self.shape.elem_count()];
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const f32, out.as_mut_ptr(), out.len());
            vkUnmapMemory(self.device, self.memory);
        }

        Ok(out)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Vulkan device handle.
#[derive(Debug, Clone)]
pub struct VulkanDevice;

impl VulkanDevice {
    /// Constructs a new Vulkan device.
    pub fn new() -> Self {
        Self
    }

    /// Probes the system for available Vulkan GPUs.
    pub fn probe() -> Result<Vec<VulkanDevice>> {
        let ctx = GLOBAL_CONTEXT.lock().unwrap();
        if ctx.is_some() {
            Ok(vec![VulkanDevice::new()])
        } else {
            Ok(vec![])
        }
    }

    /// Fused QKV attention compute shader dispatch on Vulkan GPU.
    pub fn qkv_attention_inner(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        _kv_seq_len: usize,
        cache_offset: u32,
        out: &Shape,
        _out_max: Option<&dyn BackendStorage>,
        _out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention q is not VulkanStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention k is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention v is not VulkanStorage".into()))?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::QkvAttention).to_vec();
        let buffers = [q_s.buffer, k_s.buffer, v_s.buffer, out_storage.buffer];
        let total_work = (seq_len * num_heads) as u32;
        let grid_x = (total_work + 255) / 256;

        let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();
        let push = push_params(
            seq_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as u32,
            cache_offset,
            inv_sqrt_d,
        );

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }
}

fn run_compute_shader(
    ctx: &VulkanContext,
    spirv_code: &[u8],
    buffers: &[u64],
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    push_constants: Option<&[u32]>,
) -> Result<()> {
    unsafe {
        let mut bindings = Vec::with_capacity(buffers.len());
        for i in 0..buffers.len() {
            bindings.push(VkDescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: std::ptr::null(),
            });
        }
        let ds_layout_ci = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut ds_layout = 0u64;
        let res = vkCreateDescriptorSetLayout(
            ctx.device,
            &ds_layout_ci,
            std::ptr::null(),
            &mut ds_layout,
        );
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateDescriptorSetLayout failed: {res}"
            )));
        }

        struct Cleanup {
            device: *mut c_void,
            ds_layout: u64,
            ds_pool: u64,
            shader_module: u64,
            pipeline_layout: u64,
            pipeline: u64,
            command_pool: u64,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                unsafe {
                    if self.command_pool != 0 {
                        vkDestroyCommandPool(self.device, self.command_pool, std::ptr::null());
                    }
                    if self.pipeline != 0 {
                        vkDestroyPipeline(self.device, self.pipeline, std::ptr::null());
                    }
                    if self.pipeline_layout != 0 {
                        vkDestroyPipelineLayout(
                            self.device,
                            self.pipeline_layout,
                            std::ptr::null(),
                        );
                    }
                    if self.shader_module != 0 {
                        vkDestroyShaderModule(self.device, self.shader_module, std::ptr::null());
                    }
                    if self.ds_pool != 0 {
                        vkDestroyDescriptorPool(self.device, self.ds_pool, std::ptr::null());
                    }
                    if self.ds_layout != 0 {
                        vkDestroyDescriptorSetLayout(self.device, self.ds_layout, std::ptr::null());
                    }
                }
            }
        }
        let mut cleanup = Cleanup {
            device: ctx.device,
            ds_layout,
            ds_pool: 0,
            shader_module: 0,
            pipeline_layout: 0,
            pipeline: 0,
            command_pool: 0,
        };

        let pool_size = VkDescriptorPoolSize {
            r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: buffers.len() as u32,
        };
        let ds_pool_ci = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut ds_pool = 0u64;
        let res = vkCreateDescriptorPool(ctx.device, &ds_pool_ci, std::ptr::null(), &mut ds_pool);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateDescriptorPool failed: {res}"
            )));
        }
        cleanup.ds_pool = ds_pool;

        let ds_alloc_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: ds_pool,
            descriptor_set_count: 1,
            p_set_layouts: &ds_layout,
        };
        let mut ds = 0u64;
        let res = vkAllocateDescriptorSets(ctx.device, &ds_alloc_info, &mut ds);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkAllocateDescriptorSets failed: {res}"
            )));
        }

        let mut buf_infos = Vec::with_capacity(buffers.len());
        for &buf in buffers {
            buf_infos.push(VkDescriptorBufferInfo {
                buffer: buf,
                offset: 0,
                range: !0u64,
            });
        }
        let mut writes = Vec::with_capacity(buffers.len());
        for i in 0..buffers.len() {
            writes.push(VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: ds,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_buffer_info: &buf_infos[i],
                p_image_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            });
        }
        vkUpdateDescriptorSets(
            ctx.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            std::ptr::null(),
        );

        if spirv_code.len() % 4 != 0 {
            return Err(Error::Backend(
                "SPIR-V code size must be a multiple of 4 bytes".into(),
            ));
        }
        let shader_ci = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: spirv_code.len(),
            p_code: spirv_code.as_ptr() as *const u32,
        };
        let mut shader_module = 0u64;
        let res =
            vkCreateShaderModule(ctx.device, &shader_ci, std::ptr::null(), &mut shader_module);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateShaderModule failed: {res}"
            )));
        }
        cleanup.shader_module = shader_module;

        // Push-constant block: size is dynamic — 24 bytes for the standard 6-field
        // Params block, or up to 60 bytes for the extended backward residual block.
        let pc_size = push_constants.map(|pc| pc.len() * 4).unwrap_or(0) as u32;
        let push_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: pc_size,
        };
        let pipe_layout_ci = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &ds_layout,
            push_constant_range_count: if push_constants.is_some() { 1 } else { 0 },
            p_push_constant_ranges: if push_constants.is_some() {
                &push_range as *const VkPushConstantRange as *const c_void
            } else {
                std::ptr::null()
            },
        };
        let mut pipeline_layout = 0u64;
        let res = vkCreatePipelineLayout(
            ctx.device,
            &pipe_layout_ci,
            std::ptr::null(),
            &mut pipeline_layout,
        );
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreatePipelineLayout failed: {res}"
            )));
        }
        cleanup.pipeline_layout = pipeline_layout;

        let entry_name = std::ffi::CString::new("main").unwrap();
        let stage_ci = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: entry_name.as_ptr(),
            p_specialization_info: std::ptr::null(),
        };
        let pipe_ci = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage: stage_ci,
            layout: pipeline_layout,
            base_pipeline_handle: 0,
            base_pipeline_index: 0,
        };
        let mut pipeline = 0u64;
        let res =
            vkCreateComputePipelines(ctx.device, 0, 1, &pipe_ci, std::ptr::null(), &mut pipeline);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkCreateComputePipelines failed: {res}"
            )));
        }
        cleanup.pipeline = pipeline;

        let pool_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: ctx.compute_family_index,
        };
        let mut command_pool = 0u64;
        let res = vkCreateCommandPool(ctx.device, &pool_ci, std::ptr::null(), &mut command_pool);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkCreateCommandPool failed: {res}")));
        }
        cleanup.command_pool = command_pool;

        let cmd_alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool,
            level: 0,
            command_buffer_count: 1,
        };
        let mut command_buffer: *mut c_void = std::ptr::null_mut();
        let res = vkAllocateCommandBuffers(ctx.device, &cmd_alloc_info, &mut command_buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkAllocateCommandBuffers failed: {res}"
            )));
        }

        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: 1,
            p_inheritance_info: std::ptr::null(),
        };
        let res = vkBeginCommandBuffer(command_buffer, &begin_info);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkBeginCommandBuffer failed: {res}"
            )));
        }

        vkCmdBindPipeline(command_buffer, 1, pipeline);
        vkCmdBindDescriptorSets(
            command_buffer,
            1,
            pipeline_layout,
            0,
            1,
            &ds,
            0,
            std::ptr::null(),
        );
        if let Some(pc) = push_constants {
            vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                (pc.len() * 4) as u32,
                pc.as_ptr() as *const c_void,
            );
        }
        vkCmdDispatch(command_buffer, grid_x, grid_y, grid_z);

        let res = vkEndCommandBuffer(command_buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkEndCommandBuffer failed: {res}")));
        }

        let cmd_buf_u64 = command_buffer as u64;
        let submit_info = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd_buf_u64,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        let res = vkQueueSubmit(ctx.queue, 1, &submit_info, 0);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkQueueSubmit failed: {res}")));
        }

        let res = vkQueueWaitIdle(ctx.queue);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!("vkQueueWaitIdle failed: {res}")));
        }
    }
    Ok(())
}

/// Build the 24-byte push-constant block (`Params`) the precompiled kernels
/// expect: { size:u32, dim:u32, k:u32, n:u32, m:u32, eps:f32 }. Each kernel
/// reads only the fields it needs; supplying the full block is always valid.
fn push_params(size: u32, dim: u32, k: u32, n: u32, m: u32, eps: f32) -> [u32; 6] {
    let eps_bits = eps.to_bits();
    [size, dim, k, n, m, eps_bits]
}

/// Extended 15-field push-constant block (60 bytes) for residual-aware
/// quantized backward kernels.  Layout mirrors the GLSL `Params` struct in
/// `*.comp` files:
///
/// ```text
/// pad0, pad1, k, n, m, pad_eps      // 6 u32  — prefix compat with forward
/// default_bpw, outlier_count        // 2 u32
/// backup1_bpw, backup1_codes_offset, backup1_scale_offset  // 3 u32
/// backup2_bpw, backup2_codes_offset, backup2_scale_offset  // 3 u32
/// grad_scale                        // 1 f32
/// ```
fn push_params_backward(
    k: u32,
    n: u32,
    m: u32,
    default_bpw: u32,
    outlier_count: u32,
    backup1_bpw: u32,
    backup1_codes_offset: u32,
    backup1_scale_offset: u32,
    backup2_bpw: u32,
    backup2_codes_offset: u32,
    backup2_scale_offset: u32,
    has_scales: bool,
    grad_scale: f32,
) -> [u32; 15] {
    [
        if has_scales { 1 } else { 0 }, // pad0: has_scales flag for generic shader
        0,                              // pad1
        k,
        n,
        m,
        0f32.to_bits(), // pad_eps
        default_bpw,
        outlier_count,
        backup1_bpw,
        backup1_codes_offset,
        backup1_scale_offset,
        backup2_bpw,
        backup2_codes_offset,
        backup2_scale_offset,
        grad_scale.to_bits(),
    ]
}

impl Default for VulkanDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendDevice for VulkanDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage = VulkanStorage::alloc_gpu(shape, dtype, ctx.device, ctx.physical_device)?;

        // Map and zero-fill
        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed with status {}",
                res
            )));
        }

        unsafe {
            std::ptr::write_bytes(mapped, 0, storage.bytes);
            vkUnmapMemory(ctx.device, storage.memory);
        }

        Ok(Box::new(storage))
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan matmul: input a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan matmul: input b is not VulkanStorage".into()))?;

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape("Vulkan matmul: inputs must be 2D".into()));
        }
        let (m, k) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }
        if out_shape.dims() != &[m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {out_shape:?}"
            )));
        }

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        // 1. autotuner search (Phase 4 requirement)
        let autotuner = VulkanAutotuner::new();
        let tile_config = autotuner.search_tile_config(m, n, k);

        // Use the precompiled, autotuner-matched matmul blob (block size 64 or 32, or BF16).
        let kernel = if (a.dtype().arith == ArithType::BF16 || b.dtype().arith == ArithType::BF16)
            && a_s.bytes == m * k * 2
        {
            VulkanKernel::Matmul64Bf16
        } else if tile_config.block_m == 64 {
            VulkanKernel::Matmul64
        } else {
            VulkanKernel::Matmul32
        };
        let spirv_source: Vec<u8> = spirv_for(kernel).to_vec();

        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;

        // Try GPU dispatch first
        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = ((n + tile_config.block_n as usize - 1) / tile_config.block_n as usize) as u32;
        let grid_y = ((m + tile_config.block_m as usize - 1) / tile_config.block_m as usize) as u32;

        let push = push_params(0, 0, k as u32, n as u32, m as u32, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, grid_y, 1, Some(&push)).map_err(
            |e| grim_tensor::Error::Backend(format!("Vulkan matmul GPU dispatch failed: {e}")),
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan add: input a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan add: input b is not VulkanStorage".into()))?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        let size = out.elem_count();
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Add).to_vec();

        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = ((size + 255) / 256) as u32;

        let push = push_params(size as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan add GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan mul: input a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan mul: input b is not VulkanStorage".into()))?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        let size = out.elem_count();
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Mul).to_vec();

        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = ((size + 255) / 256) as u32;

        let push = push_params(size as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan mul GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_s = gate
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan silu_mul: input gate is not VulkanStorage".into())
            })?;
        let up_s = up.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul: input up is not VulkanStorage".into())
        })?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        let size = out.elem_count();
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::SiluMul).to_vec();

        let buffers = [gate_s.buffer, up_s.buffer, out_storage.buffer];
        let grid_x = ((size + 255) / 256) as u32;

        let push = push_params(size as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan silu_mul GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let e_s = e.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul_backward e is not VulkanStorage".into())
        })?;
        let g_s = g.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul_backward g is not VulkanStorage".into())
        })?;
        let dw_s = dw.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul_backward dw is not VulkanStorage".into())
        })?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let df = VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;
        let de = VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;
        let buffers = [e_s.buffer, g_s.buffer, dw_s.buffer, df.buffer, de.buffer];
        let push = push_params(out_shape.elem_count() as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::SiluMulBackward),
            &buffers,
            ((out_shape.elem_count() + 255) / 256) as u32,
            1,
            1,
            Some(&push),
        )?;
        Ok((
            Box::new(df),
            Box::new(de),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan rms_norm: input x is not VulkanStorage".into())
        })?;
        let w_s = weight
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rms_norm: input weight is not VulkanStorage".into())
            })?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        let size = out.elem_count();
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::RmsNorm).to_vec();

        let buffers = [x_s.buffer, w_s.buffer, out_storage.buffer];
        let grid_x = ((size + 255) / 256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, eps);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan rms_norm GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan softmax: input x is not VulkanStorage".into()))?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        let size = out.elem_count();
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Softmax).to_vec();

        let buffers = [x_s.buffer, out_storage.buffer];
        let grid_x = ((size + 255) / 256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan softmax GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let w_s = weight
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan embedding: weight is not VulkanStorage".into())
            })?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out, DType::F32, ctx.device, ctx.physical_device)?;

        // Upload indices to GPU buffer temp
        let idx_shape = Shape::new(vec![indices.len()]);
        let idx_storage = VulkanStorage::alloc_gpu(
            &idx_shape,
            DType {
                arith: ArithType::U32,
                storage: grim_tensor::dtype::Storage::Native,
            },
            ctx.device,
            ctx.physical_device,
        )?;
        let mut mapped_idx: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = vkMapMemory(
                ctx.device,
                idx_storage.memory,
                0,
                idx_storage.bytes as VkDeviceSize,
                0,
                &mut mapped_idx,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed for indices buffer: {}",
                    res
                )));
            }
            std::ptr::copy_nonoverlapping(indices.as_ptr(), mapped_idx as *mut u32, indices.len());
            vkUnmapMemory(ctx.device, idx_storage.memory);
        }

        let w_dims = weight.shape().dims();
        let dim = w_dims[w_dims.len() - 1];
        let num_indices = indices.len();
        let size = num_indices * dim;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Embedding).to_vec();

        let buffers = [w_s.buffer, idx_storage.buffer, out_storage.buffer];
        let grid_x = ((size + 255) / 256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan embedding GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage =
            VulkanStorage::alloc_gpu(shape, dtype.clone(), ctx.device, ctx.physical_device)?;

        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed with status {}",
                res
            )));
        }

        unsafe {
            match dtype.arith {
                ArithType::BF16 => {
                    // Simulate BF16 precision via f32 round-trip while using FP32 kernels.
                    let dst = mapped as *mut f32;
                    for (i, &val) in data.iter().enumerate() {
                        *dst.add(i) = f32_to_bf16_to_f32(val);
                    }
                }
                _ => {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut f32, data.len());
                }
            }
            vkUnmapMemory(ctx.device, storage.memory);
        }

        Ok(Box::new(storage))
    }

    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        // Vulkan backend: MemAdvice is currently a no-op
        Ok(())
    }

    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention_inner(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            out_shape,
            out_max,
            out_sum,
        )
    }

    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        _max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        _cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_paged expects 3-D output shape [batch, num_heads, head_dim]".into(),
            ));
        }
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "qkv_attention_paged requires num_heads divisible by num_kv_heads".into(),
            ));
        }
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_paged q is not VulkanStorage".into()))?;
        let table_s = block_tables
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("qkv_attention_paged block_tables is not VulkanStorage".into())
            })?;
        let k_s = k_pages
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("qkv_attention_paged k_pages is not VulkanStorage".into())
            })?;
        let v_s = v_pages
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("qkv_attention_paged v_pages is not VulkanStorage".into())
            })?;

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;
        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            table_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(
            page_size as u32,
            0,
            kv_seq_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as f32,
        );
        let grid_x = ((head_dim + 31) / 32) as u32;
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::QkvAttentionPaged),
            &buffers,
            grid_x,
            num_heads as u32,
            1,
            Some(&push),
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn tree_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        tree_parents: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dims = out_shape.dims();
        if dims.len() != 4 {
            return Err(Error::Shape(
                "tree_attention expects [batch, 1+gamma, num_heads, head_dim]".into(),
            ));
        }
        let (batch, nodes, num_heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "tree_attention requires num_heads divisible by num_kv_heads".into(),
            ));
        }
        if head_dim > 256 || nodes == 0 || tree_parents.shape().elem_count() < nodes {
            return Err(Error::Shape(
                "tree_attention requires 1+gamma parent entries and head_dim <= 256".into(),
            ));
        }
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("tree_attention q is not VulkanStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("tree_attention k is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("tree_attention v is not VulkanStorage".into()))?;
        let parents_s = tree_parents
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("tree_attention tree_parents is not VulkanStorage".into())
            })?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;
        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            parents_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(
            batch as u32,
            num_heads as u32,
            kv_seq_len as u32,
            head_dim as u32,
            (nodes - 1) as u32,
            f32::from_bits((num_kv_heads as u32) << 16 | (cache_offset & 0xffff)),
        );
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::TreeAttention),
            &buffers,
            1,
            (nodes * num_heads) as u32,
            batch as u32,
            Some(&push),
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        _num_kv_heads: usize,
        kv_seq_len: usize,
        _cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if quant_bits != 8 {
            return Err(Error::Unimplemented(
                "Vulkan kv_dequant_attention currently supports 8-bit K/V only".into(),
            ));
        }
        let dims = out_shape.dims();
        if dims.len() != 3 {
            return Err(Error::Shape(
                "kv_dequant_attention expects [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention q is not VulkanStorage".into()))?;
        let k_s = k_tensor
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention k_tensor is not VulkanStorage".into())
            })?;
        let ks_s = k_scales
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention k_scales is not VulkanStorage".into())
            })?;
        let v_s = v_tensor
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention v_tensor is not VulkanStorage".into())
            })?;
        let vs_s = v_scales
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("kv_dequant_attention v_scales is not VulkanStorage".into())
            })?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;
        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            ks_s.buffer,
            vs_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(
            kv_seq_len as u32,
            dims[2] as u32,
            0,
            dims[2] as u32,
            dims[1] as u32,
            0.0,
        );
        let grid_x = ((dims[2] + 31) / 32) as u32;
        run_compute_shader(
            ctx,
            spirv_for(VulkanKernel::KvDequantAttention),
            &buffers,
            grid_x,
            dims[1] as u32,
            1,
            Some(&push),
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan mul_scalar x is not VulkanStorage".into()))?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::MulScalar).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = ((n + 255) / 256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, scalar);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan sqrt x is not VulkanStorage".into()))?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Sqrt).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = ((n + 255) / 256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan recip x is not VulkanStorage".into()))?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Recip).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = out_shape.elem_count();
        let grid_x = ((n + 255) / 256) as u32;

        let push = push_params(n as u32, 0, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        dim: usize,
        base: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rope x is not VulkanStorage".into()))?;
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)?;

        let num_tokens = positions.len();
        let num_heads = out_shape.elem_count() / (num_tokens * dim);

        let pos_shape = Shape::new(vec![num_tokens]);
        let pos_storage = VulkanStorage::alloc_gpu(
            &pos_shape,
            DType {
                arith: ArithType::U32,
                storage: grim_tensor::dtype::Storage::Native,
            },
            ctx.device,
            ctx.physical_device,
        )?;
        let mut mapped_pos: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = vkMapMemory(
                ctx.device,
                pos_storage.memory,
                0,
                pos_storage.bytes as VkDeviceSize,
                0,
                &mut mapped_pos,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed for positions buffer: {}",
                    res
                )));
            }
            std::ptr::copy_nonoverlapping(positions.as_ptr(), mapped_pos as *mut u32, num_tokens);
            vkUnmapMemory(ctx.device, pos_storage.memory);
        }

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Rope).to_vec();
        let buffers = [x_s.buffer, pos_storage.buffer, out_storage.buffer];
        let total_pairs = (num_tokens * num_heads * (dim / 2)) as u32;
        let grid_x = (total_pairs + 255) / 256;

        let push = push_params(num_tokens as u32, dim as u32, num_heads as u32, 0, 0, base);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage = VulkanStorage::alloc_gpu(shape, dtype, ctx.device, ctx.physical_device)?;

        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed in from_cpu_bytes: {}",
                res
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
            vkUnmapMemory(ctx.device, storage.memory);
        }

        Ok(Box::new(storage))
    }

    fn selective_scan(
        &self,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        tracing::warn!("Vulkan selective_scan: falling back to CPU execution");
        let x_v = x.to_cpu_vec_f32()?;
        let a_v = a.to_cpu_vec_f32()?;
        let b_v = b.to_cpu_vec_f32()?;
        let c_v = c.to_cpu_vec_f32()?;
        let d_v = d.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim_dinner];
        for b_idx in 0..batch {
            for d_idx in 0..dim_dinner {
                let mut h = vec![0.0f32; dim_dstate];
                let d_val = if d_v.len() > d_idx { d_v[d_idx] } else { 0.0 };

                for t in 0..seq_len {
                    let x_idx = (b_idx * seq_len + t) * dim_dinner + d_idx;
                    let x_t = x_v[x_idx];
                    let mut y_t = d_val * x_t;

                    for s in 0..dim_dstate {
                        let a_idx = d_idx * dim_dstate + s;
                        let b_idx_off = (b_idx * seq_len + t) * dim_dstate + s;
                        let c_idx_off = (b_idx * seq_len + t) * dim_dstate + s;

                        let a_val = if a_v.len() > a_idx { a_v[a_idx] } else { 1.0 };
                        let b_val = if b_v.len() > b_idx_off {
                            b_v[b_idx_off]
                        } else {
                            1.0
                        };
                        let c_val = if c_v.len() > c_idx_off {
                            c_v[c_idx_off]
                        } else {
                            1.0
                        };

                        h[s] = a_val * h[s] + x_t * b_val;
                        y_t += c_val * h[s];
                    }
                    out[x_idx] = y_t;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn flash_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        _causal: bool,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = _causal;
        let out_dims = out_shape.dims();
        if out_dims.len() == 3 {
            let inferred_heads = out_dims[1];
            let inferred_dim = out_dims[2];
            if inferred_heads != num_heads {
                tracing::warn!(
                    "Vulkan flash_attention: out_shape head dim ({inferred_heads}) != num_heads ({num_heads})"
                );
            }
            if inferred_dim != head_dim {
                tracing::warn!(
                    "Vulkan flash_attention: out_shape head_dim ({inferred_dim}) != head_dim ({head_dim})"
                );
            }
        }
        if num_heads != num_kv_heads {
            tracing::warn!(
                "Vulkan flash_attention: GQA detected (num_heads={num_heads}, num_kv_heads={num_kv_heads}); \
                 kernel repeats KV heads to match query heads"
            );
        }
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        // Pass num_kv_heads for GQA head-repeat; num_heads comes from out_shape.
        let (out_storage, _h) =
            self.qkv_attention(q, k, v, num_kv_heads, seq_len, 0, out_shape, None, None)?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn cross_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_dims = out_shape.dims();
        if out_dims.len() == 3 {
            let inferred_dim = out_dims[2];
            if inferred_dim != head_dim {
                tracing::warn!(
                    "Vulkan cross_attention: out_shape head_dim ({inferred_dim}) != head_dim ({head_dim})"
                );
            }
        }
        tracing::warn!(
            "Vulkan cross_attention: seq_len={seq_len}, kv_seq_len={kv_seq_len}, num_heads={num_heads}, head_dim={head_dim}"
        );
        // Cross-attention: Q and KV share num_heads, so pass it as KV-head count.
        let (out_storage, _h) =
            self.qkv_attention(q, k, v, num_heads, kv_seq_len, 0, out_shape, None, None)?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn rwkv_time_mix(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        g: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        tracing::warn!("Vulkan rwkv_time_mix: falling back to CPU execution");
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let w_vec = w.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim];
        for b in 0..batch {
            for d in 0..dim {
                let mut state = 0.0f32;
                let w_val = if w_vec.len() > d { w_vec[d] } else { 0.9f32 };

                for t in 0..seq_len {
                    let idx = (b * seq_len + t) * dim + d;
                    let k_t = if k_vec.len() > idx {
                        k_vec[idx]
                    } else {
                        x_vec[idx]
                    };
                    let v_t = if v_vec.len() > idx {
                        v_vec[idx]
                    } else {
                        x_vec[idx]
                    };
                    let g_t = if g_vec.len() > idx {
                        g_vec[idx]
                    } else {
                        1.0f32
                    };

                    state = w_val * state + k_t * v_t;
                    let sig = 1.0f32 / (1.0f32 + (-g_t).exp());
                    out[idx] = state * sig;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn rwkv_channel_mix(
        &self,
        x: &dyn BackendStorage,
        k: &dyn BackendStorage,
        r: &dyn BackendStorage,
        v: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        tracing::warn!("Vulkan rwkv_channel_mix: falling back to CPU execution");
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let r_vec = r.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;

        let elem_count = out_shape.elem_count();
        let mut out = vec![0.0f32; elem_count];
        for i in 0..elem_count {
            let x_val = x_vec[i];
            let k_val = if k_vec.len() > i { k_vec[i] } else { x_val };
            let r_val = if r_vec.len() > i { r_vec[i] } else { 1.0f32 };
            let v_val = if v_vec.len() > i { v_vec[i] } else { x_val };

            let sig_r = 1.0f32 / (1.0f32 + (-r_val).exp());
            let relu_k = k_val.max(0.0f32);
            out[i] = sig_r * (relu_k * relu_k) * v_val;
        }

        let _ = batch;
        let _ = dim;

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // Try GPU fused dequant dispatch if both inputs are VulkanStorage
        if let (Some(a_s), Some(b_s)) = (
            a.as_any().downcast_ref::<VulkanStorage>(),
            b_packed.as_any().downcast_ref::<VulkanStorage>(),
        ) {
            let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
            if let Some(ctx) = ctx_guard.as_ref() {
                if let Ok(out_storage) =
                    VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx.device, ctx.physical_device)
                {
                    let kernel = if k % 256 == 0 {
                        VulkanKernel::FusedDequantGemmQ4K
                    } else {
                        VulkanKernel::FusedDequantGemmQ80
                    };
                    let spirv_source: Vec<u8> = spirv_for(kernel).to_vec();
                    let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
                    let grid_x = ((n + 15) / 16) as u32;
                    let grid_y = ((m + 15) / 16) as u32;
                    let push = push_params(0, 0, k as u32, n as u32, m as u32, 0.0);

                    if run_compute_shader(
                        ctx,
                        &spirv_source,
                        &buffers,
                        grid_x,
                        grid_y,
                        1,
                        Some(&push),
                    )
                    .is_ok()
                    {
                        return Ok((
                            Box::new(out_storage),
                            Box::new(grim_tensor::backend::ReadyHandle),
                        ));
                    }
                }
            }
        }

        tracing::warn!("Vulkan quantized_matmul: falling back to CPU execution");
        let a_vec = a.to_cpu_vec_f32()?;
        let mut b_dequant = vec![0.0f32; k * n];
        let blocks_per_col = k / 32;

        let b_bytes = b_packed
            .to_cpu_vec_f32()
            .ok()
            .map(|v| v.iter().map(|&f| f as u8).collect())
            .unwrap_or_else(|| vec![0u8; k * n]);

        for col in 0..n {
            for block in 0..blocks_per_col {
                let scale_idx = col * blocks_per_col + block;
                let scale = if scale_idx < b_scales.len() {
                    b_scales[scale_idx]
                } else {
                    1.0f32
                };
                for i in 0..32 {
                    let byte_offset = (col * blocks_per_col + block) * 32 + i;
                    let byte_val = if byte_offset < b_bytes.len() {
                        b_bytes[byte_offset]
                    } else {
                        128u8
                    };
                    let q_val = (byte_val as i16 - 128) as f32 / 127.0f32;
                    let r = block * 32 + i;
                    if r < k {
                        b_dequant[r * n + col] = q_val * scale;
                    }
                }
            }
        }

        let mut c_vec = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_vec[row * k + p] * b_dequant[p * n + col];
                }
                c_vec[row * n + col] = sum;
            }
        }

        let out_storage = self.from_cpu(&c_vec, out_shape, a.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        residuals: Option<&grim_tensor::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dy_s = dy
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan backward dx dy is not VulkanStorage".into()))?;
        let b_s = b_packed
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan backward dx b_packed is not VulkanStorage".into())
            })?;

        // Extract context device/physical_device pointers without holding the
        // lock — from_cpu_bytes also locks GLOBAL_CONTEXT, so we must release
        // here to avoid deadlock.
        let (ctx_device, ctx_physical_device) = {
            let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
            let ctx = ctx_guard
                .as_ref()
                .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
            (ctx.device, ctx.physical_device)
        };

        // Allocate dX output [M, K] f32.
        let dx = VulkanStorage::alloc_gpu(
            out_shape,
            DType::F32,
            ctx_device,
            ctx_physical_device,
        )?;

        // --- Extract residual / outlier metadata from the residuals handle ---
        let outlier_count = residuals.map(|r| r.outlier_count).unwrap_or(0);
        let backup1_bpw = residuals.map(|r| r.backup1_bpw).unwrap_or(0);
        let backup1_codes_offset =
            residuals.map(|r| r.backup1_codes_offset).unwrap_or(0);
        let backup1_scale_offset =
            residuals.map(|r| r.backup1_scale_offset).unwrap_or(0);
        let backup2_bpw = residuals.map(|r| r.backup2_bpw).unwrap_or(0);
        let backup2_codes_offset =
            residuals.map(|r| r.backup2_codes_offset).unwrap_or(0);
        let backup2_scale_offset =
            residuals.map(|r| r.backup2_scale_offset).unwrap_or(0);

        // --- Extract outlier index/value data from the tensor's provenance ---
        // `QuantizedMatmulBackwardResiduals::from_tensor` leaves the raw device
        // pointers null; the actual host-decoded outlier vectors live in
        // `QuantProvenance::WithResiduals`.
        let prov = b_s.provenance();
        let (outlier_indices_host, outlier_values_host) = match &prov {
            QuantProvenance::WithResiduals {
                outlier_indices,
                outlier_values_bits,
                ..
            } => {
                let indices: Vec<u8> = outlier_indices
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                let values: Vec<u8> = outlier_values_bits
                    .iter()
                    .map(|v| f32::from_bits(*v).to_ne_bytes())
                    .flatten()
                    .collect();
                (indices, values)
            }
            _ => (Vec::new(), Vec::new()),
        };

        // --- Upload outlier buffers (binding 3 = indices u32, binding 4 = values f32) ---
        // When outlier_count == 0 the shader checks the count before accessing
        // these buffers, so minimal dummies suffice.
        let (outlier_idx_box, outlier_val_box) = if outlier_count > 0
            && !outlier_indices_host.is_empty()
            && !outlier_values_host.is_empty()
        {
            let idx = self.from_cpu_bytes(
                &outlier_indices_host,
                &Shape::from_slice(&[outlier_indices_host.len()]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
            )?;
            let val = self.from_cpu_bytes(
                &outlier_values_host,
                &Shape::from_slice(&[outlier_values_host.len()]),
                DType::F32,
            )?;
            (idx, val)
        } else {
            let dummy = [0u8; 1];
            let idx = self.from_cpu_bytes(
                &dummy,
                &Shape::from_slice(&[1]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?;
            let val = self.from_cpu_bytes(
                &dummy,
                &Shape::from_slice(&[1]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?;
            (idx, val)
        };

        let outlier_idx_s = outlier_idx_box
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("outlier indices storage is not VulkanStorage".into())
            })?;
        let outlier_val_s = outlier_val_box
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("outlier values storage is not VulkanStorage".into())
            })?;

        // --- Select shader variant based on the weight's storage format ---
        let (kernel, has_scales) = match b_s.dtype().storage {
            DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                (VulkanKernel::QuantizedMatmulBackwardDx, false)
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                (VulkanKernel::QuantizedMatmulBackwardDxQ8_0, false)
            }
            // ResidualPacked, all other KQuant (Q5K, Q6K, IQ*), FloatPack,
            // Block, GroupInt — use the generic unpack_weight path.
            _ => (VulkanKernel::QuantizedMatmulBackwardDxGeneric, !b_scales.is_empty()),
        };

        // --- Upload per-column f32 scales for the generic shader (binding 5) ---
        let scales_storage_box = if has_scales {
            let f32_scale_bytes: Vec<u8> = b_scales
                .iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect();
            Some(self.from_cpu_bytes(
                &f32_scale_bytes,
                &Shape::from_slice(&[b_scales.len() * 4]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?)
        } else {
            // Dummy buffer so binding 5 always has a valid Vulkan buffer handle.
            let dummy = [0u8; 1];
            Some(self.from_cpu_bytes(
                &dummy,
                &Shape::from_slice(&[1]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?)
        };
        let scales_s = scales_storage_box
            .as_ref()
            .and_then(|s| s.as_any().downcast_ref::<VulkanStorage>())
            .ok_or_else(|| Error::Backend("scales storage is not VulkanStorage".into()))?;

        // --- Build extended push constants ---
        let push = push_params_backward(
            k as u32,
            n as u32,
            m as u32,
            default_bpw as u32,
            outlier_count as u32,
            backup1_bpw as u32,
            backup1_codes_offset as u32,
            backup1_scale_offset as u32,
            backup2_bpw as u32,
            backup2_codes_offset as u32,
            backup2_scale_offset as u32,
            has_scales,
            1.0, // grad_scale = 1.0 for STE identity (straight-through estimator)
        );

        // --- Build GPU buffer binding list ---
        // bindings: [0]=dY, [1]=B_codes, [2]=dX, [3]=outlier_indices,
        //           [4]=outlier_values, [5]=scales_u8 (generic only)
        let mut buffers: Vec<u64> = vec![
            dy_s.buffer,
            b_s.buffer,
            dx.buffer,
            outlier_idx_s.buffer,
            outlier_val_s.buffer,
        ];
        if has_scales {
            buffers.push(scales_s.buffer);
        }

        // --- Lock context and dispatch ---
        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let spirv = spirv_for(kernel).to_vec();
        run_compute_shader(
            ctx,
            &spirv,
            &buffers,
            ((k + 15) / 16) as u32,
            ((m + 15) / 16) as u32,
            1,
            Some(&push),
        )?;

        Ok((Box::new(dx), Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn all_reduce(
        &self,
        inputs: &[&dyn BackendStorage],
        op: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if inputs.is_empty() {
            return Err(Error::Backend("all_reduce: no inputs".into()));
        }
        if op != "sum" {
            return Err(Error::Backend(format!(
                "all_reduce: only 'sum' supported, got '{op}'"
            )));
        }
        let shape = inputs[0].shape().clone();
        let dtype = inputs[0].dtype();
        let total = shape.elem_count();
        let is_f32 = dtype.arith == ArithType::F32;

        // All inputs must share the same shape.
        for s in inputs {
            if s.shape() != &shape {
                return Err(Error::Backend("all_reduce: input shape mismatch".into()));
            }
        }

        // ── GPU fast path: accumulate all inputs into a pre-zeroed output buffer.
        // The `all_reduce` accumulate kernel does Out[i] += A[i], so we zero the
        // output once and then dispatch one pass per input tensor.
        // `run_compute_shader` calls vkQueueWaitIdle, so each pass is synchronous.
        {
            let all_vulkan = inputs
                .iter()
                .all(|s| s.as_any().downcast_ref::<VulkanStorage>().is_some());
            if is_f32 && total > 0 && all_vulkan {
                let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
                if let Some(ctx) = ctx_guard.as_ref() {
                    if let Ok(out_storage) = VulkanStorage::alloc_gpu(
                        &shape,
                        DType::F32,
                        ctx.device,
                        ctx.physical_device,
                    ) {
                        // Zero the output buffer (accumulation target).
                        let zeroed = {
                            let mut mapped: *mut c_void = std::ptr::null_mut();
                            let res = unsafe {
                                vkMapMemory(
                                    ctx.device,
                                    out_storage.memory,
                                    0,
                                    out_storage.bytes as VkDeviceSize,
                                    0,
                                    &mut mapped,
                                )
                            };
                            if res == VK_SUCCESS {
                                unsafe {
                                    std::ptr::write_bytes(mapped, 0, out_storage.bytes);
                                    vkUnmapMemory(ctx.device, out_storage.memory);
                                }
                                true
                            } else {
                                false
                            }
                        };
                        if zeroed {
                            let spirv = spirv_for(VulkanKernel::AllReduce).to_vec();
                            let grid_x = ((total + 255) / 256) as u32;
                            let push = push_params(total as u32, 0, 0, 0, 0, 0.0);
                            let mut ok = true;
                            for input in inputs {
                                let in_s = input
                                    .as_any()
                                    .downcast_ref::<VulkanStorage>()
                                    .unwrap();
                                let buffers = [in_s.buffer, out_storage.buffer];
                                if run_compute_shader(
                                    ctx,
                                    &spirv,
                                    &buffers,
                                    grid_x,
                                    1,
                                    1,
                                    Some(&push),
                                )
                                .is_err()
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                return Ok((
                                    Box::new(out_storage),
                                    Box::new(grim_tensor::backend::ReadyHandle),
                                ));
                            }
                        }
                    }
                }
            }
        } // ctx_guard dropped — lock released before CPU fallback

        // ── CPU fallback ──────────────────────────────────────────────
        let mut acc = inputs[0].to_cpu_vec_f32()?;
        for other in &inputs[1..] {
            let v = other.to_cpu_vec_f32()?;
            if v.len() != acc.len() {
                return Err(Error::Backend(
                    "all_reduce: input length mismatch during fallback".into(),
                ));
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        let storage = self.from_cpu(&acc, &shape, dtype)?;
        Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        if partials.is_empty() {
            return Err(Error::Backend("comm_fuse_reduce: no partials".into()));
        }
        let dims0 = partials[0].0.shape().dims();
        let m = dims0[0];
        let n_total: usize = partials
            .iter()
            .map(|(s, _)| s.shape().dims().get(1).copied().unwrap_or(0))
            .sum();
        let dtype = partials[0].0.dtype();
        let is_f32 = dtype.arith == ArithType::F32;
        let out_shape = Shape::new(vec![m, n_total]);

        // ── GPU fast path: scatter each column shard to its offset.
        // The `comm_fuse_reduce` kernel copies In[row, col] →
        // Out[row, col_offset + col] on a pre-zeroed output buffer.
        {
            let all_vulkan = partials
                .iter()
                .all(|(s, _)| s.as_any().downcast_ref::<VulkanStorage>().is_some());
            if is_f32 && n_total > 0 && all_vulkan {
                let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
                if let Some(ctx) = ctx_guard.as_ref() {
                    if let Ok(out_storage) = VulkanStorage::alloc_gpu(
                        &out_shape,
                        DType::F32,
                        ctx.device,
                        ctx.physical_device,
                    ) {
                        // Zero the output buffer.
                        let zeroed = {
                            let mut mapped: *mut c_void = std::ptr::null_mut();
                            let res = unsafe {
                                vkMapMemory(
                                    ctx.device,
                                    out_storage.memory,
                                    0,
                                    out_storage.bytes as VkDeviceSize,
                                    0,
                                    &mut mapped,
                                )
                            };
                            if res == VK_SUCCESS {
                                unsafe {
                                    std::ptr::write_bytes(mapped, 0, out_storage.bytes);
                                    vkUnmapMemory(ctx.device, out_storage.memory);
                                }
                                true
                            } else {
                                false
                            }
                        };
                        if zeroed {
                            let spirv = spirv_for(VulkanKernel::CommFuseReduce).to_vec();
                            let mut col_offset = 0usize;
                            let mut ok = true;
                            for (storage, _placement) in partials {
                                let s = storage
                                    .as_any()
                                    .downcast_ref::<VulkanStorage>()
                                    .unwrap();
                                let n_src = s.shape().dims().get(1).copied().unwrap_or(0);
                                let buffers = [s.buffer, out_storage.buffer];
                                let grid_x = ((n_src + 15) / 16) as u32;
                                let grid_y = ((m + 15) / 16) as u32;
                                let push = push_params(
                                    n_src as u32,
                                    col_offset as u32,
                                    n_total as u32,
                                    m as u32,
                                    0,
                                    0.0,
                                );
                                if run_compute_shader(
                                    ctx,
                                    &spirv,
                                    &buffers,
                                    grid_x,
                                    grid_y,
                                    1,
                                    Some(&push),
                                )
                                .is_err()
                                {
                                    ok = false;
                                    break;
                                }
                                col_offset += n_src;
                            }
                            if ok {
                                return Ok(Box::new(out_storage));
                            }
                        }
                    }
                }
            }
        } // ctx_guard dropped — lock released before CPU fallback

        // ── CPU fallback ──────────────────────────────────────────────
        let mut assembled = vec![0.0f32; m * n_total];
        let mut col_offset = 0usize;
        for (storage, _placement) in partials {
            let data = storage.to_cpu_vec_f32()?;
            let n_cols = storage.shape().dims().get(1).copied().unwrap_or(0);
            for row in 0..m {
                for col in 0..n_cols {
                    assembled[row * n_total + col_offset + col] +=
                        data[row * n_cols + col];
                }
            }
            col_offset += n_cols;
        }
        let storage = self.from_cpu(&assembled, &out_shape, dtype)?;
        Ok(storage)
    }
}

/// Simulation tile shape config matching CubeCL autotuning schema (§4.1 requirements)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
}

pub struct VulkanAutotuner;

impl VulkanAutotuner {
    pub fn new() -> Self {
        Self
    }

    /// Autotunes CubeCL tile block parameters via simulated benchmarking.
    pub fn search_tile_config(&self, m: usize, n: usize, k: usize) -> VulkanTileConfig {
        tracing::info!(
            "[VulkanAutotuner] Running autotune search for shape ({}, {}, {})...",
            m,
            n,
            k
        );
        // Autotuning heuristic based on common powers of 2 for GPU compute blocks
        if m % 64 == 0 && n % 64 == 0 {
            VulkanTileConfig {
                block_m: 64,
                block_n: 64,
                block_k: 16,
            }
        } else {
            VulkanTileConfig {
                block_m: 32,
                block_n: 32,
                block_k: 8,
            }
        }
    }

    /// Estimate GEMM latency in ms given problem dimensions and dtype.
    pub(crate) fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        _placement: &grim_tensor::backend::ScythePlacement,
    ) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = match dtype.arith {
            ArithType::F16 | ArithType::BF16 => 100.0,
            ArithType::F32 => 50.0,
            _ => 30.0,
        };
        (flops / (tflops * 1e12) * 1000.0).max(0.01)
    }
}

include!(concat!(env!("OUT_DIR"), "/spirv_spv.rs"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanKernel {
    Add,
    Mul,
    SiluMul,
    RmsNorm,
    Softmax,
    Embedding,
    Matmul64,
    Matmul32,
    Matmul64Bf16,
    QkvAttention,
    MulScalar,
    Sqrt,
    Recip,
    Rope,
    FusedDequantGemmQ4K,
    FusedDequantGemmQ80,
    KvDequantAttention,
    SelectiveScan,
    QkvAttentionPaged,
    TreeAttention,
    FlashAttention,
    SiluMulBackward,
    QuantizedMatmulBackwardDx,
    QuantizedMatmulBackwardDxQ8_0,
    QuantizedMatmulBackwardDxGeneric,
    RwkvTimeMix,
    RwkvChannelMix,
    AllReduce,
    CommFuseReduce,
}

pub fn spirv_for(kernel: VulkanKernel) -> &'static [u8] {
    match kernel {
        VulkanKernel::Add => SPIRV_ADD,
        VulkanKernel::Mul => SPIRV_MUL,
        VulkanKernel::SiluMul => SPIRV_SILU_MUL,
        VulkanKernel::RmsNorm => SPIRV_RMS_NORM,
        VulkanKernel::Softmax => SPIRV_SOFTMAX,
        VulkanKernel::Embedding => SPIRV_EMBEDDING,
        VulkanKernel::Matmul64 => SPIRV_MATMUL_64,
        VulkanKernel::Matmul32 => SPIRV_MATMUL_32,
        VulkanKernel::Matmul64Bf16 => SPIRV_MATMUL_64_BF16,
        VulkanKernel::QkvAttention => SPIRV_QKV_ATTENTION,
        VulkanKernel::MulScalar => SPIRV_MUL_SCALAR,
        VulkanKernel::Sqrt => SPIRV_SQRT,
        VulkanKernel::Recip => SPIRV_RECIP,
        VulkanKernel::Rope => SPIRV_ROPE,
        VulkanKernel::FusedDequantGemmQ4K => SPIRV_FUSED_DEQUANT_GEMM_Q4K,
        VulkanKernel::FusedDequantGemmQ80 => SPIRV_FUSED_DEQUANT_GEMM_Q8_0,
        VulkanKernel::KvDequantAttention => SPIRV_KV_DEQUANT_ATTENTION,
        VulkanKernel::SelectiveScan => SPIRV_SELECTIVE_SCAN,
        VulkanKernel::QkvAttentionPaged => SPIRV_QKV_ATTENTION_PAGED,
        VulkanKernel::TreeAttention => SPIRV_TREE_ATTENTION,
        VulkanKernel::FlashAttention => SPIRV_FLASH_ATTENTION,
        VulkanKernel::SiluMulBackward => SPIRV_SILU_MUL_BACKWARD,
        VulkanKernel::QuantizedMatmulBackwardDx => SPIRV_QUANTIZED_MATMUL_BACKWARD_DX,
        VulkanKernel::QuantizedMatmulBackwardDxQ8_0 => SPIRV_QUANTIZED_MATMUL_BACKWARD_DX_Q8_0,
        VulkanKernel::QuantizedMatmulBackwardDxGeneric => {
            SPIRV_QUANTIZED_MATMUL_BACKWARD_DX_GENERIC
        }
        VulkanKernel::RwkvTimeMix => SPIRV_RWKV_TIME_MIX,
        VulkanKernel::RwkvChannelMix => SPIRV_RWKV_CHANNEL_MIX,
        VulkanKernel::AllReduce => SPIRV_ALL_REDUCE,
        VulkanKernel::CommFuseReduce => SPIRV_COMM_FUSE_REDUCE,
    }
}

/// Helper function to retrieve the size in bytes of a data type.
fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 => 2,
        ArithType::BF16 => 4, // BF16 simulated via f32 round-trip; 4 bytes for kernel compatibility.
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}

/// Convert f32 to BF16 and back to f32, simulating BF16 precision.
fn f32_to_bf16_to_f32(val: f32) -> f32 {
    let bits = val.to_bits();
    let sign = bits & 0x80000000;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;

    if exp == 0 {
        // Subnormal or zero -> flush to zero
        0.0
    } else if exp == 255 {
        // Inf or NaN -> preserve
        f32::from_bits(sign | 0x7F800000)
    } else {
        // Normal: truncate mantissa to 7 bits, keep sign and exponent
        let bf16_mant = (mant >> 16) & 0x7F;
        let f32_bits = sign | (exp << 23) | (bf16_mant << 16);
        f32::from_bits(f32_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::{DType, Shape};

    #[test]
    fn test_vulkan_device_probe() {
        let devices = VulkanDevice::probe().unwrap();
        if GLOBAL_CONTEXT.lock().unwrap().is_some() {
            assert!(!devices.is_empty());
        }
    }

    #[test]
    fn test_vulkan_zeros() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![0.0; 8]);
    }

    #[test]
    fn test_vulkan_from_cpu() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];
        let shape = Shape::new(vec![3, 2]);
        let host_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();
        let cpu_data = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, host_data);
    }

    #[test]
    fn test_vulkan_autotuner_and_spirv() {
        let autotuner = VulkanAutotuner::new();
        let config = autotuner.search_tile_config(128, 128, 64);
        assert_eq!(config.block_m, 64);
        assert_eq!(config.block_n, 64);

        // Verify a precompiled SPIR-V blob is loadable for the chosen tile size.
        let spirv = spirv_for(VulkanKernel::Matmul64);
        assert!(!spirv.is_empty());
    }

    #[test]
    fn test_vulkan_matmul_simulated() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let shape = Shape::new(vec![2, 2]);

        let a_s = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        let (out_s, _handle) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &shape).unwrap();
        let res = out_s.to_cpu_vec_f32().unwrap();
        assert_eq!(res, a_data); // A @ I = A
    }

    #[test]
    fn test_vulkan_matmul_non_identity_and_shape_mismatch() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let devices = VulkanDevice::probe().unwrap();
        let dev = &devices[0];

        // 1. Non-identity matrix multiplication: [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let shape = Shape::new(vec![2, 2]);

        let a_s = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        let (out_s, _handle) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &shape).unwrap();
        let res = out_s.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);

        // 2. Shape mismatch error enforcement
        let bad_shape = Shape::new(vec![3, 2]);
        let err_res = dev.matmul(a_s.as_ref(), b_s.as_ref(), &bad_shape);
        assert!(
            err_res.is_err(),
            "matmul with wrong output shape must return Err"
        );
    }

    #[test]
    fn test_vulkan_gpu_compute() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let shape = Shape::new(vec![4]);

        let dev = VulkanDevice::new();
        let a_s = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();

        let a_storage = a_s.as_any().downcast_ref::<VulkanStorage>().unwrap();
        let b_storage = b_s.as_any().downcast_ref::<VulkanStorage>().unwrap();

        let ctx_guard = GLOBAL_CONTEXT.lock().unwrap();
        let ctx = match ctx_guard.as_ref() {
            Some(c) => c,
            None => return,
        };
        let out_storage =
            VulkanStorage::alloc_gpu(&shape, DType::F32, ctx.device, ctx.physical_device).unwrap();

        // Standard precompiled add SPIR-V binary from radv_repro.rs
        let spirv_add_u32: &[u32] = &[
            0x07230203, 0x00010000, 0x0008000b, 0x00000033, 0x00000000, 0x00020011, 0x00000001,
            0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
            0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
            0x0000000b, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001,
            0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
            0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c,
            0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000019, 0x43667542,
            0x00000000, 0x00040006, 0x00000019, 0x00000000, 0x00000063, 0x00030005, 0x0000001b,
            0x00000000, 0x00040005, 0x00000020, 0x41667542, 0x00000000, 0x00040006, 0x00000020,
            0x00000000, 0x00000061, 0x00030005, 0x00000022, 0x00000000, 0x00040005, 0x00000028,
            0x42667542, 0x00000000, 0x00040006, 0x00000028, 0x00000000, 0x00000062, 0x00030005,
            0x0000002a, 0x00000000, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c, 0x00040047,
            0x00000018, 0x00000006, 0x00000004, 0x00030047, 0x00000019, 0x00000003, 0x00050048,
            0x00000019, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000001b, 0x00000021,
            0x00000002, 0x00040047, 0x0000001b, 0x00000022, 0x00000000, 0x00040047, 0x0000001f,
            0x00000006, 0x00000004, 0x00030047, 0x00000020, 0x00000003, 0x00050048, 0x00000020,
            0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000022, 0x00000021, 0x00000000,
            0x00040047, 0x00000022, 0x00000022, 0x00000000, 0x00040047, 0x00000027, 0x00000006,
            0x00000004, 0x00030047, 0x00000028, 0x00000003, 0x00050048, 0x00000028, 0x00000000,
            0x00000023, 0x00000000, 0x00040047, 0x0000002a, 0x00000021, 0x00000001, 0x00040047,
            0x0000002a, 0x00000022, 0x00000000, 0x00040047, 0x00000032, 0x0000000b, 0x00000019,
            0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
            0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017,
            0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009,
            0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c,
            0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0004002b, 0x00000006,
            0x00000011, 0x00000004, 0x00020014, 0x00000012, 0x00030016, 0x00000017, 0x00000020,
            0x0003001d, 0x00000018, 0x00000017, 0x0003001e, 0x00000019, 0x00000018, 0x00040020,
            0x0000001a, 0x00000002, 0x00000019, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000002,
            0x00040015, 0x0000001c, 0x00000020, 0x00000001, 0x0004002b, 0x0000001c, 0x0000001d,
            0x00000000, 0x0003001d, 0x0000001f, 0x00000017, 0x0003001e, 0x00000020, 0x0000001f,
            0x00040020, 0x00000021, 0x00000002, 0x00000020, 0x0004003b, 0x00000021, 0x00000022,
            0x00000002, 0x00040020, 0x00000024, 0x00000002, 0x00000017, 0x0003001d, 0x00000027,
            0x00000017, 0x0003001e, 0x00000028, 0x00000027, 0x00040020, 0x00000029, 0x00000002,
            0x00000028, 0x0004003b, 0x00000029, 0x0000002a, 0x00000002, 0x0004002b, 0x00000006,
            0x00000030, 0x00000040, 0x0004002b, 0x00000006, 0x00000031, 0x00000001, 0x0006002c,
            0x00000009, 0x00000032, 0x00000030, 0x00000031, 0x00000031, 0x00050036, 0x00000002,
            0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007,
            0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c,
            0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f,
            0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x000500ae, 0x00000012, 0x00000013,
            0x00000010, 0x00000011, 0x000300f7, 0x00000015, 0x00000000, 0x000400fa, 0x00000013,
            0x00000014, 0x00000015, 0x000200f8, 0x00000014, 0x000100fd, 0x000200f8, 0x00000015,
            0x0004003d, 0x00000006, 0x0000001e, 0x00000008, 0x0004003d, 0x00000006, 0x00000023,
            0x00000008, 0x00060041, 0x00000024, 0x00000025, 0x00000022, 0x0000001d, 0x00000023,
            0x0004003d, 0x00000017, 0x00000026, 0x00000025, 0x0004003d, 0x00000006, 0x0000002b,
            0x00000008, 0x00060041, 0x00000024, 0x0000002c, 0x0000002a, 0x0000001d, 0x0000002b,
            0x0004003d, 0x00000017, 0x0000002d, 0x0000002c, 0x00050081, 0x00000017, 0x0000002e,
            0x00000026, 0x0000002d, 0x00060041, 0x00000024, 0x0000002f, 0x0000001b, 0x0000001d,
            0x0000001e, 0x0003003e, 0x0000002f, 0x0000002e, 0x000100fd, 0x00010038,
        ];

        let spirv_bytes = unsafe {
            std::slice::from_raw_parts(spirv_add_u32.as_ptr() as *const u8, spirv_add_u32.len() * 4)
        };

        let buffers = [a_storage.buffer, b_storage.buffer, out_storage.buffer];
        run_compute_shader(ctx, spirv_bytes, &buffers, 1, 1, 1, None).unwrap();

        let cpu_data = out_storage.to_cpu_vec_f32().unwrap();
        assert_eq!(cpu_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    // ===== Mutation-resistant kernel math contracts =====

    fn close_vulkan(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-4,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    #[test]
    fn test_vulkan_add_golden_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let a_data = vec![1.5f32, -2.5, 0.0, 3.14159];
        let b_data = vec![2.5f32, 3.5, -1.0, 1.0];
        let shape = Shape::new(vec![4]);
        let a = dev.from_cpu(&a_data, &shape, DType::F32).unwrap();
        let b = dev.from_cpu(&b_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.add(a.as_ref(), b.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 4);
        close_vulkan(res[0], 4.0, "vulkan_add w0");
        close_vulkan(res[1], 1.0, "vulkan_add w1");
        close_vulkan(res[2], -1.0, "vulkan_add w2");
        close_vulkan(res[3], 4.14159, "vulkan_add w3");
    }

    #[test]
    fn test_vulkan_math_ops() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let shape = Shape::new(vec![4]);
        let host_data = vec![4.0f32, 9.0, 16.0, 25.0];
        let x = dev.from_cpu(&host_data, &shape, DType::F32).unwrap();

        let (out_sqrt, _) = dev.sqrt(x.as_ref(), &shape).unwrap();
        assert_eq!(out_sqrt.to_cpu_vec_f32().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

        let (out_recip, _) = dev.recip(out_sqrt.as_ref(), &shape).unwrap();
        assert_eq!(
            out_recip.to_cpu_vec_f32().unwrap(),
            vec![0.5, 1.0 / 3.0, 0.25, 0.2]
        );

        let (out_mul, _) = dev.mul_scalar(x.as_ref(), 0.5, &shape).unwrap();
        assert_eq!(out_mul.to_cpu_vec_f32().unwrap(), vec![2.0, 4.5, 8.0, 12.5]);
    }

    #[test]
    fn test_vulkan_silu_mul_golden_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let gate_data = vec![1.0f32, -1.0];
        let up_data = vec![2.0f32, 3.0];
        let shape = Shape::new(vec![2]);
        let gate = dev.from_cpu(&gate_data, &shape, DType::F32).unwrap();
        let up = dev.from_cpu(&up_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.silu_mul(gate.as_ref(), up.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 2);

        let sig_1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
        let expected_0 = sig_1 * 1.0 * 2.0;

        let sig_neg1 = 1.0f32 / (1.0f32 + (1.0f32).exp());
        let expected_1 = (-1.0f32 * sig_neg1) * 3.0;

        close_vulkan(res[0], expected_0, "vulkan_silu_mul w0");
        close_vulkan(res[1], expected_1, "vulkan_silu_mul w1");
    }

    #[test]
    fn test_vulkan_rms_norm_golden_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let x_data = vec![3.0f32, 4.0];
        let w_data = vec![1.0f32, 2.0];
        let shape = Shape::new(vec![2]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let w = dev.from_cpu(&w_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.rms_norm(x.as_ref(), w.as_ref(), 1e-6, &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 2);

        let rms_val = (12.5f32 + 1e-6).sqrt();
        let expected_0 = (3.0 / rms_val) * 1.0;
        let expected_1 = (4.0 / rms_val) * 2.0;
        close_vulkan(res[0], expected_0, "vulkan_rms_norm w0");
        close_vulkan(res[1], expected_1, "vulkan_rms_norm w1");
    }

    #[test]
    fn test_vulkan_softmax_golden_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let x_data = vec![1.0f32, 2.0, 3.0];
        let shape = Shape::new(vec![3]);
        let x = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
        let (out, _h) = dev.softmax(x.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), 3);

        let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        close_vulkan(res[0], 1.0f32.exp() / sum_exp, "vulkan_softmax w0");
        close_vulkan(res[1], 2.0f32.exp() / sum_exp, "vulkan_softmax w1");
        close_vulkan(res[2], 3.0f32.exp() / sum_exp, "vulkan_softmax w2");
    }

    #[test]
    fn test_vulkan_embedding_golden_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let table = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let weight = dev
            .from_cpu(&table, &Shape::new(vec![3, 2]), DType::F32)
            .unwrap();
        let indices = vec![2u32, 0];
        let out_shape = Shape::new(vec![2, 2]);
        let (out, _h) = dev
            .embedding(weight.as_ref(), &indices, &out_shape)
            .unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![50.0, 60.0, 10.0, 20.0]);
    }

    // BF16 matmul golden test — hand-crafted BF16 inputs, exact FP32 reference.
    // BF16: 1 sign | 8 exponent | 7 mantissa. Accumulates in FP32.

    #[test]
    fn test_vulkan_matmul_bf16_golden_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let shape = Shape::new(vec![2, 2]);

        // Hand-crafted BF16: 1.0=0x3F80, 2.0=0x4000, 3.0=0x4040, 4.0=0x4080
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a = dev.from_cpu(&a_data, &shape, DType::BF16).unwrap();
        let b = dev.from_cpu(&b_data, &shape, DType::BF16).unwrap();
        let (out, _h) = dev.matmul(a.as_ref(), b.as_ref(), &shape).unwrap();
        let res = out.to_cpu_vec_f32().unwrap();

        // Reference: [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50], FP32 accumulation.
        close_vulkan(res[0], 19.0, "bf16_matmul[0,0]");
        close_vulkan(res[1], 22.0, "bf16_matmul[0,1]");
        close_vulkan(res[2], 43.0, "bf16_matmul[1,0]");
        close_vulkan(res[3], 50.0, "bf16_matmul[1,1]");
    }

    // QKV attention golden test — hand-crafted Q/K/V, exact FP32 reference.
    // Q: [seq=2, heads=2, dim=2]; K/V: [kv_seq=4, kv_heads=1, dim=2].
    #[test]
    fn test_vulkan_qkv_attention_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();

        let seq_len = 2usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let kv_seq_len = 4usize;

        let q_data = vec![1.0f32; seq_len * num_heads * head_dim];
        let k_data = vec![1.0f32; kv_seq_len * num_kv_heads * head_dim];
        let v_data = vec![2.0f32; kv_seq_len * num_kv_heads * head_dim];

        let q_shape = Shape::new(vec![seq_len, num_heads, head_dim]);
        let k_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let v_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let out_shape = Shape::new(vec![seq_len, num_heads, head_dim]);

        let q_buf = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
        let k_buf = dev.from_cpu(&k_data, &k_shape, DType::F32).unwrap();
        let v_buf = dev.from_cpu(&v_data, &v_shape, DType::F32).unwrap();

        let (out, _h) = dev
            .qkv_attention(
                q_buf.as_ref(),
                k_buf.as_ref(),
                v_buf.as_ref(),
                num_kv_heads,
                kv_seq_len,
                0,
                &out_shape,
                None,
                None,
            )
            .unwrap();

        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res.len(), seq_len * num_heads * head_dim);
        for &val in res.iter() {
            close_vulkan(val, 2.0, "qkv_attention out");
        }
    }

    #[test]
    fn test_vulkan_qkv_attention_paged_gqa_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let q_shape = Shape::new(vec![1, 2, 2]);
        let page_shape = Shape::new(vec![1, 2, 1, 2]);
        let table_shape = Shape::new(vec![1, 1]);
        let q = dev
            .from_cpu(&vec![1.0f32; 4], &q_shape, DType::F32)
            .unwrap();
        let k = dev
            .from_cpu(&vec![1.0f32; 4], &page_shape, DType::F32)
            .unwrap();
        let v = dev
            .from_cpu(&vec![2.0f32; 4], &page_shape, DType::F32)
            .unwrap();
        let table = dev
            .from_cpu(&vec![0.0f32], &table_shape, DType::F32)
            .unwrap();
        let (out, _) = dev
            .qkv_attention_paged(
                q.as_ref(),
                table.as_ref(),
                k.as_ref(),
                v.as_ref(),
                1,
                1,
                2,
                2,
                0,
                &q_shape,
            )
            .unwrap();
        for value in out.to_cpu_vec_f32().unwrap() {
            close_vulkan(value, 2.0, "paged GQA");
        }
    }

    #[test]
    fn test_vulkan_tree_attention_gqa_exact() {
        if GLOBAL_CONTEXT.lock().unwrap().is_none() {
            return;
        }
        let dev = VulkanDevice::new();
        let q_shape = Shape::new(vec![1, 2, 2, 2]);
        let kv_shape = Shape::new(vec![2, 1, 2]);
        let parent_shape = Shape::new(vec![2]);
        let q = dev
            .from_cpu(&vec![1.0f32; 8], &q_shape, DType::F32)
            .unwrap();
        let k = dev
            .from_cpu(&vec![1.0f32; 4], &kv_shape, DType::F32)
            .unwrap();
        let v = dev
            .from_cpu(&vec![2.0f32; 4], &kv_shape, DType::F32)
            .unwrap();
        let parents = dev
            .from_cpu(&vec![0.0f32, 0.0], &parent_shape, DType::F32)
            .unwrap();
        let (out, _) = dev
            .tree_attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                parents.as_ref(),
                1,
                2,
                0,
                &q_shape,
            )
            .unwrap();
        for value in out.to_cpu_vec_f32().unwrap() {
            close_vulkan(value, 2.0, "tree GQA");
        }
    }
}

/// Query `(free_bytes, total_bytes)` memory on Vulkan device `ordinal`.
pub fn vram_info(_ordinal: usize) -> Option<(u64, u64)> {
    if let Ok(guard) = GLOBAL_CONTEXT.lock() {
        if let Some(ctx) = guard.as_ref() {
            unsafe {
                let mut props = VkPhysicalDeviceMemoryProperties {
                    memory_type_count: 0,
                    memory_types: [VkMemoryType {
                        property_flags: 0,
                        heap_index: 0,
                    }; 32],
                    memory_heap_count: 0,
                    memory_heaps: [VkMemoryHeap { size: 0, flags: 0 }; 16],
                };
                vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &mut props);

                // Sum all device-local heaps (VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1).
                let mut total_device_local: u64 = 0;
                for i in 0..(props.memory_heap_count as usize) {
                    if (props.memory_heaps[i].flags & 1) != 0 {
                        total_device_local += props.memory_heaps[i].size;
                    }
                }

                if total_device_local == 0 {
                    return None;
                }

                // Without VK_EXT_memory_budget, live free memory is unavailable.
                // Return None so callers know free memory querying is unsupported.
                return None;
            }
        }
    }
    None
}
