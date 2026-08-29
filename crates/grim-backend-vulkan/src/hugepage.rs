//! 2MB Linux HugePage Host Memory for Vulkan Zero-Copy DMA Transfers.
//!
//! Allocates 2MB-aligned host memory via `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` on Linux
//! (with automatic fallback to aligned anonymous memory) for low-latency host-to-device
//! and device-to-host memory transport.

use std::ptr::NonNull;
use grim_tensor::error::{Error, Result};

const HUGEPAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB

/// Host-pinned/hugepage buffer for Vulkan host staging.
pub struct VulkanHugePageBuffer {
    ptr: NonNull<u8>,
    size: usize,
    is_hugepage: bool,
}

unsafe impl Send for VulkanHugePageBuffer {}
unsafe impl Sync for VulkanHugePageBuffer {}

impl VulkanHugePageBuffer {
    /// Allocate a new buffer of at least `requested_size` bytes rounded up to 2MB.
    pub fn new(requested_size: usize) -> Result<Self> {
        let size = (requested_size + HUGEPAGE_SIZE - 1) & !(HUGEPAGE_SIZE - 1);
        let size = if size == 0 { HUGEPAGE_SIZE } else { size };

        let (ptr, is_hugepage) = Self::alloc_mmap(size)?;

        Ok(Self {
            ptr,
            size,
            is_hugepage,
        })
    }

    #[cfg(target_os = "linux")]
    fn alloc_mmap(size: usize) -> Result<(NonNull<u8>, bool)> {
        let flags_huge = libc::MAP_PRIVATE
            | libc::MAP_ANONYMOUS
            | libc::MAP_HUGETLB
            | (21 << libc::MAP_HUGE_SHIFT);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                flags_huge,
                -1,
                0,
            )
        };

        if ptr != libc::MAP_FAILED {
            if let Some(non_null) = NonNull::new(ptr as *mut u8) {
                return Ok((non_null, true));
            }
        }

        // Fallback: standard anonymous aligned mmap
        let flags_anon = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                flags_anon,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(Error::Backend(format!(
                "mmap failed to allocate {} bytes: {}",
                size,
                std::io::Error::last_os_error()
            )));
        }

        NonNull::new(ptr as *mut u8)
            .map(|p| (p, false))
            .ok_or_else(|| Error::Backend("mmap returned null pointer".into()))
    }

    #[cfg(not(target_os = "linux"))]
    fn alloc_mmap(size: usize) -> Result<(NonNull<u8>, bool)> {
        let layout = std::alloc::Layout::from_size_align(size, HUGEPAGE_SIZE)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        NonNull::new(ptr)
            .map(|p| (p, false))
            .ok_or_else(|| Error::Backend("allocation failed".into()))
    }

    /// Size of the allocated buffer in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Whether this buffer is backed by 2MB OS Hugepages.
    pub fn is_hugepage(&self) -> bool {
        self.is_hugepage
    }

    /// Raw pointer to buffer memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable raw pointer to buffer memory.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for VulkanHugePageBuffer {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        unsafe {
            let _ = libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size);
        }

        #[cfg(not(target_os = "linux"))]
        unsafe {
            if let Ok(layout) = std::alloc::Layout::from_size_align(self.size, HUGEPAGE_SIZE) {
                std::alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}
