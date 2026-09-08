//! Vulkan storage: GPU buffer management, host-visible staging, read-back.

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendStorage, Shape};

use crate::context::{QUEUE_LOCK, global_context};
use crate::dtype_byte_size;
use crate::ffi::*;

// Vulkan crate structs

/// A handle to a Vulkan compute operation. INVARIANT: `run_compute_shader` calls `vkQueueWaitIdle` synchronously during dispatch.
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
    pub(crate) buffer: u64,
    pub(crate) memory: u64,
    pub(crate) bytes: usize,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) provenance: QuantProvenance,
    pub(crate) device: *mut c_void,
    /// Whether the backing `memory` is host-visible. Device-local buffers
    /// cannot be `vkMapMemory`'d and are read back via a staging copy.
    pub(crate) host_visible: bool,
}

unsafe impl Send for VulkanStorage {}
unsafe impl Sync for VulkanStorage {}

/// Which memory tier `alloc_gpu_inner` should prefer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuMemoryTier {
    /// Require HOST_VISIBLE | HOST_COHERENT (mappable; used for uploads).
    HostVisible,
    /// Prefer DEVICE_LOCAL, falling back to host-visible when no device-local
    /// type matches the buffer's `memory_type_bits`.
    DeviceLocal,
}

impl VulkanStorage {
    /// Allocates memory and a buffer on the Vulkan device (host-visible).
    pub fn alloc_gpu(
        shape: &Shape,
        dtype: DType,
        device: *mut c_void,
        physical_device: *mut c_void,
    ) -> Result<Self> {
        Self::alloc_gpu_inner(
            shape,
            dtype,
            device,
            physical_device,
            GpuMemoryTier::HostVisible,
        )
    }

    /// Allocates a buffer preferring `DEVICE_LOCAL` VRAM for compute outputs, falling back to a host-visible type where no suitable device-local type exists (e.g.
    /// some UMA/APU configs).
    pub fn alloc_device_local_gpu(
        shape: &Shape,
        dtype: DType,
        device: *mut c_void,
        physical_device: *mut c_void,
    ) -> Result<Self> {
        Self::alloc_gpu_inner(
            shape,
            dtype,
            device,
            physical_device,
            GpuMemoryTier::DeviceLocal,
        )
    }

    fn alloc_gpu_inner(
        shape: &Shape,
        dtype: DType,
        device: *mut c_void,
        physical_device: *mut c_void,
        tier: GpuMemoryTier,
    ) -> Result<Self> {
        let bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype))
            .ok_or_else(|| {
                Error::Backend(format!(
                    "alloc_gpu: byte count overflow for shape {:?} dtype {:?}",
                    shape, dtype
                ))
            })?;

        let alloc_bytes = bytes.max(16);
        let buffer_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size: alloc_bytes as VkDeviceSize,
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

        // Select a memory type for the requested tier.
        // HostVisible requires a mappable+coherent type; DeviceLocal prefers VRAM and falls back to a mappable type.
        let (memory_type_index, host_visible) = {
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

            let mappable =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            let find = |required: u32| -> Option<u32> {
                (0..mem_properties.memory_type_count).find(|i| {
                    (reqs.memory_type_bits & (1 << i)) != 0
                        && (mem_properties.memory_types[*i as usize].property_flags & required)
                            == required
                })
            };

            match tier {
                GpuMemoryTier::HostVisible => (
                    find(mappable).ok_or_else(|| {
                        Error::Backend("Failed to find suitable Vulkan memory type".into())
                    })?,
                    true,
                ),
                GpuMemoryTier::DeviceLocal => match find(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                    // A device-local type that also happens to be mappable
                    // (UMA) can still be read directly.
                    Some(i) => {
                        let flags = mem_properties.memory_types[i as usize].property_flags;
                        (i, (flags & mappable) == mappable)
                    }
                    None => (
                        find(mappable).ok_or_else(|| {
                            Error::Backend("Failed to find suitable Vulkan memory type".into())
                        })?,
                        true,
                    ),
                },
            }
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
            // This allocator records the tier actually selected above.
            host_visible,
        })
    }

    /// Read the raw backing bytes, routing device-local buffers through a staging copy.
    /// Prefer this over direct `vkMapMemory` for readback so the caller works regardless of which memory.
    pub(crate) fn read_raw_bytes(&self) -> Result<Vec<u8>> {
        if self.host_visible {
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
            let bytes = unsafe {
                let slice = std::slice::from_raw_parts(mapped as *const u8, self.bytes);
                let v = slice.to_vec();
                vkUnmapMemory(self.device, self.memory);
                v
            };
            Ok(bytes)
        } else {
            // Device-local buffers are not host-mappable: route through a staging buffer copy on the compute queue.
            // `read_back_via_staging` acquires the global context itself, so callers must not hold the context lock (BackendStorage.
            read_back_via_staging(self)
        }
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
        let raw = self.read_raw_bytes()?;
        let expected = self
            .shape
            .elem_count()
            .checked_mul(4)
            .ok_or_else(|| Error::Backend("to_cpu_vec_f32: elem_count overflow".into()))?;
        if raw.len() < expected {
            return Err(Error::Backend(format!(
                "to_cpu_vec_f32: read {} bytes, expected at least {}",
                raw.len(),
                expected
            )));
        }
        // Safe byte-to-f32 reinterpretation: no raw pointer cast.
        // A Vec<u8>'s backing allocation is only 1-byte aligned by Rust guarantees, so casting `raw.as_ptr()` to.
        Ok(raw
            .chunks_exact(4)
            .take(self.shape.elem_count())
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Dtype-aware u32 readback: U32 buffers are chunked straight from bytes (native layout - no f32 reinterpretation garbage),
    /// I64 truncates through `i64`, everything else falls back to the f32 path + cast (F32-backed scratch tensors).
    fn to_cpu_vec_u32(&self) -> Result<Vec<u32>> {
        match self.dtype.arith {
            ArithType::U32 => {
                let raw = self.read_raw_bytes()?;
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            ArithType::I64 => {
                let raw = self.read_raw_bytes()?;
                Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32
                    })
                    .collect())
            }
            _ => Ok(self
                .to_cpu_vec_f32()?
                .into_iter()
                .map(|v| v as u32)
                .collect()),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Allocate a host-visible, host-coherent staging buffer on `device`.
/// Returns `(buffer, memory)`. The caller owns cleanup.
fn alloc_host_visible_staging_buffer(
    device: *mut c_void,
    physical_device: *mut c_void,
    bytes: usize,
) -> Result<(u64, u64)> {
    unsafe {
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
        let res = vkCreateBuffer(device, &buffer_ci, std::ptr::null(), &mut buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "alloc_host_visible_staging_buffer: vkCreateBuffer failed: {res}"
            )));
        }

        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        vkGetBufferMemoryRequirements(device, buffer, &mut reqs);

        let mut mem_properties = VkPhysicalDeviceMemoryProperties {
            memory_type_count: 0,
            memory_types: [VkMemoryType {
                property_flags: 0,
                heap_index: 0,
            }; 32],
            memory_heap_count: 0,
            memory_heaps: [VkMemoryHeap { size: 0, flags: 0 }; 16],
        };
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mut mem_properties);

        let mappable = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let memory_type_index = (0..mem_properties.memory_type_count)
            .find(|i| {
                (reqs.memory_type_bits & (1 << i)) != 0
                    && (mem_properties.memory_types[*i as usize].property_flags & mappable)
                        == mappable
            })
            .ok_or_else(|| Error::Backend("staging: no mappable memory type".into()))?;

        let alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: reqs.size,
            memory_type_index,
        };

        let mut memory: u64 = 0;
        let res = vkAllocateMemory(device, &alloc_info, std::ptr::null(), &mut memory);
        if res != VK_SUCCESS {
            vkDestroyBuffer(device, buffer, std::ptr::null());
            return Err(Error::Backend(format!(
                "alloc_host_visible_staging_buffer: vkAllocateMemory failed: {res}"
            )));
        }

        let res = vkBindBufferMemory(device, buffer, memory, 0);
        if res != VK_SUCCESS {
            vkFreeMemory(device, memory, std::ptr::null());
            vkDestroyBuffer(device, buffer, std::ptr::null());
            return Err(Error::Backend(format!(
                "alloc_host_visible_staging_buffer: vkBindBufferMemory failed: {res}"
            )));
        }

        Ok((buffer, memory))
    }
}

/// Synchronously copy `size` bytes from `src_buffer` (device) into `dst_buffer` (host-visible staging) using a one-shot command buffer on the compute queue.
/// Compute queues support transfer operations.
fn copy_device_buffer_to_host(
    device: *mut c_void,
    queue: *mut c_void,
    compute_family_index: u32,
    src_buffer: u64,
    dst_buffer: u64,
    size: u64,
) -> Result<()> {
    unsafe {
        let pool_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: compute_family_index,
        };
        let mut command_pool = 0u64;
        let res = vkCreateCommandPool(device, &pool_ci, std::ptr::null(), &mut command_pool);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_device_buffer_to_host: vkCreateCommandPool failed: {res}"
            )));
        }
        struct PoolCleanup {
            device: *mut c_void,
            command_pool: u64,
        }
        impl Drop for PoolCleanup {
            fn drop(&mut self) {
                if self.command_pool != 0 {
                    unsafe {
                        vkDestroyCommandPool(self.device, self.command_pool, std::ptr::null());
                    }
                }
            }
        }
        let _pool = PoolCleanup {
            device,
            command_pool,
        };

        let cmd_alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool,
            level: 0,
            command_buffer_count: 1,
        };
        let mut command_buffer: *mut c_void = std::ptr::null_mut();
        let res = vkAllocateCommandBuffers(device, &cmd_alloc_info, &mut command_buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_device_buffer_to_host: vkAllocateCommandBuffers failed: {res}"
            )));
        }

        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: 1, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            p_inheritance_info: std::ptr::null(),
        };
        let res = vkBeginCommandBuffer(command_buffer, &begin_info);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_device_buffer_to_host: vkBeginCommandBuffer failed: {res}"
            )));
        }

        let region = VkBufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &region);

        let res = vkEndCommandBuffer(command_buffer);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_device_buffer_to_host: vkEndCommandBuffer failed: {res}"
            )));
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
        let _q_lock = QUEUE_LOCK.lock().unwrap();
        let res = vkQueueSubmit(queue, 1, &submit_info, 0);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_device_buffer_to_host: vkQueueSubmit failed: {res}"
            )));
        }
        let res = vkQueueWaitIdle(queue);
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "copy_device_buffer_to_host: vkQueueWaitIdle failed: {res}"
            )));
        }
    }
    Ok(())
}

/// Read back a device-local `VulkanStorage` by copying into a host-visible staging buffer.
/// Acquires the global context for the compute queue; callers must NOT hold the context lock.
fn read_back_via_staging(storage: &VulkanStorage) -> Result<Vec<u8>> {
    let (device, queue, compute_family_index, physical_device) = {
        let guard = global_context();
        let ctx = guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        (
            ctx.device,
            ctx.queue,
            ctx.compute_family_index,
            ctx.physical_device,
        )
    };
    let (staging_buffer, staging_memory) =
        alloc_host_visible_staging_buffer(device, physical_device, storage.bytes)?;

    struct StagingCleanup {
        device: *mut c_void,
        buffer: u64,
        memory: u64,
    }
    impl Drop for StagingCleanup {
        fn drop(&mut self) {
            unsafe {
                if self.memory != 0 {
                    vkFreeMemory(self.device, self.memory, std::ptr::null());
                }
                if self.buffer != 0 {
                    vkDestroyBuffer(self.device, self.buffer, std::ptr::null());
                }
            }
        }
    }
    let _staging = StagingCleanup {
        device,
        buffer: staging_buffer,
        memory: staging_memory,
    };

    copy_device_buffer_to_host(
        device,
        queue,
        compute_family_index,
        storage.buffer,
        staging_buffer,
        storage.bytes as u64,
    )?;

    let mut mapped: *mut c_void = std::ptr::null_mut();
    let res = unsafe {
        vkMapMemory(
            device,
            staging_memory,
            0,
            storage.bytes as VkDeviceSize,
            0,
            &mut mapped,
        )
    };
    if res != VK_SUCCESS {
        return Err(Error::Backend(format!(
            "read_back_via_staging: vkMapMemory failed with status {}",
            res
        )));
    }
    let bytes = unsafe {
        let slice = std::slice::from_raw_parts(mapped as *const u8, storage.bytes);
        let v = slice.to_vec();
        vkUnmapMemory(device, staging_memory);
        v
    };
    Ok(bytes)
}
