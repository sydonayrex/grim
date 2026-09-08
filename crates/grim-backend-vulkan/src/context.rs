//! Vulkan context: device initialization, queue management, pipeline setup.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use grim_tensor::error::{Error, Result};

use crate::ffi::*;

// Vulkan helper context

pub(crate) struct VulkanContext {
    pub(crate) instance: *mut c_void,
    pub(crate) physical_device: *mut c_void,
    pub(crate) device: *mut c_void,
    pub(crate) queue: *mut c_void,
    pub(crate) compute_family_index: u32,
    pub(crate) device_name: String,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) driver_version: u32,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
    fn init() -> Result<Self> {
        // Do NOT enable third-party layers (e.g. Steam overlay, MangoHud, Bumblebee);
        // they hang headless environments. Disable implicit layers unless user specified otherwise.
        if std::env::var("VK_LOADER_LAYERS_DISABLE").is_err() {
            unsafe {
                std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~all~");
            }
        }

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
        let res =
            unsafe { vkEnumeratePhysicalDevices(instance, &mut gpu_count, gpus.as_mut_ptr()) };
        if res != VK_SUCCESS || gpus.is_empty() || gpus.iter().all(|&p| p.is_null()) {
            unsafe {
                vkDestroyInstance(instance, std::ptr::null());
            }
            return Err(Error::Backend(format!(
                "vkEnumeratePhysicalDevices failed with status {}",
                res
            )));
        }

        // Iterate devices and choose best GPU (prefer discrete GPU over integrated GPU; reject CPU).
        let mut chosen_dev = None;
        let mut chosen_props = None;

        for &dev in &gpus {
            if dev.is_null() {
                continue;
            }
            let mut props = VkPhysicalDeviceProperties {
                api_version: 0,
                driver_version: 0,
                vendor_id: 0,
                device_id: 0,
                device_type: 0,
                device_name: [0u8; 256],
            };
            unsafe { vkGetPhysicalDeviceProperties(dev, &mut props) };
            if props.device_type == VK_PHYSICAL_DEVICE_TYPE_CPU {
                continue;
            }
            if props.device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
                chosen_dev = Some(dev);
                chosen_props = Some(props);
                break;
            }
            if chosen_dev.is_none() {
                chosen_dev = Some(dev);
                chosen_props = Some(props);
            }
        }

        let (physical_device, props) = match (chosen_dev, chosen_props) {
            (Some(d), Some(p)) => (d, p),
            _ => {
                unsafe {
                    vkDestroyInstance(instance, std::ptr::null());
                }
                return Err(Error::Backend(
                    "No valid non-CPU Vulkan GPU device found".into(),
                ));
            }
        };

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
            return Err(Error::Backend(
                "vkGetDeviceQueue returned null queue pointer".into(),
            ));
        }

        Ok(Self {
            instance,
            physical_device,
            device,
            queue,
            compute_family_index,
            device_name: read_device_name(&props.device_name),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            driver_version: props.driver_version,
        })
    }
}

/// Convert a null-terminated `VkPhysicalDeviceProperties.device_name`
/// (`[u8; 256]`) into a `String`.
fn read_device_name(name: &[u8; 256]) -> String {
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    String::from_utf8_lossy(&name[..end]).into_owned()
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
    pub(crate) static ref QUEUE_LOCK: Mutex<()> = Mutex::new(());
}

/// Guards against re-attempting Vulkan init on every consumer call after a persistent failure.
/// A single on-demand retry (see `global_context`) is enough; re-running init in a hot loop would.
static RETRY_ATTEMPTED: AtomicBool = AtomicBool::new(false);

/// Re-initializes the global Vulkan context after a failed init.
/// `lazy_static` caches `None` forever when the initial `VulkanContext::init` fails (e.g.
pub fn reset_global_context() -> Result<()> {
    let mut guard = GLOBAL_CONTEXT.lock().unwrap();
    if guard.is_none() {
        RETRY_ATTEMPTED.store(true, Ordering::SeqCst);
        *guard = VulkanContext::init().ok();
    }
    if guard.is_some() {
        Ok(())
    } else {
        Err(Error::Backend(
            "Vulkan context re-initialization failed".into(),
        ))
    }
}

/// Accessor for the global context that re-attempts init once when the initial `lazy_static` init failed (which would otherwise cache `None` forever).
/// A persistent failure is not re-tried on every call thanks to `RETRY_ATTEMPTED`; callers see `None`.
pub(crate) fn global_context() -> std::sync::MutexGuard<'static, Option<VulkanContext>> {
    let mut guard = GLOBAL_CONTEXT.lock().unwrap();
    if guard.is_none() && !RETRY_ATTEMPTED.swap(true, Ordering::SeqCst) {
        *guard = VulkanContext::init().ok();
    }
    guard
}
