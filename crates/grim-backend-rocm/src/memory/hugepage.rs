//! 2MB Linux HugePage Host-Pinned Memory Allocator.
//!
//! Provides `HugePagePinnedBuffer`, which allocates 2MB-aligned host memory
//! via `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` on Linux (falling back to aligned
//! anonymous mmap on platforms/systems without hugepages configured) and
//! registers it with the ROCm driver via `hipHostRegister`.
//!
//! This enables maximum DMA throughput over PCIe Gen4/Gen5 links by eliminating
//! 4KB TLB miss penalties and scatter-gather lists.

use std::ffi::c_void;
use std::ptr::NonNull;
use grim_tensor::error::{Error, Result};
use crate::device::handles::{hipHostRegister, hipHostUnregister};

const HUGEPAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB

/// Host-pinned buffer backed by 2MB hugepages when available.
pub struct HugePagePinnedBuffer {
    ptr: NonNull<u8>,
    size: usize,
    is_hugepage: bool,
    is_registered: bool,
}

unsafe impl Send for HugePagePinnedBuffer {}
unsafe impl Sync for HugePagePinnedBuffer {}

impl HugePagePinnedBuffer {
    /// Allocate a new host-pinned buffer of at least `size` bytes rounded to 2MB.
    pub fn new(requested_size: usize) -> Result<Self> {
        let size = (requested_size + HUGEPAGE_SIZE - 1) & !(HUGEPAGE_SIZE - 1);
        let size = if size == 0 { HUGEPAGE_SIZE } else { size };

        let (ptr, is_hugepage) = Self::alloc_mmap(size)?;

        // Register host memory with HIP for zero-copy DMA access
        let res = unsafe { hipHostRegister(ptr.as_ptr() as *mut c_void, size, 0) };
        let is_registered = res == 0;
        if !is_registered {
            // Note: In non-GPU test environments or when HIP driver is unavailable,
            // the buffer remains usable as high-speed hugepage host memory.
        }

        Ok(Self {
            ptr,
            size,
            is_hugepage,
            is_registered,
        })
    }

    #[cfg(target_os = "linux")]
    fn alloc_mmap(size: usize) -> Result<(NonNull<u8>, bool)> {
        // Try 2MB Hugepages first
        let flags_huge = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | (21 << libc::MAP_HUGE_SHIFT);
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

    /// Whether this buffer is backed by OS 2MB Hugepages.
    pub fn is_hugepage(&self) -> bool {
        self.is_hugepage
    }

    /// Whether this buffer is registered with HIP for GPU DMA.
    pub fn is_registered(&self) -> bool {
        self.is_registered
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

impl Drop for HugePagePinnedBuffer {
    fn drop(&mut self) {
        if self.is_registered {
            unsafe {
                let _ = hipHostUnregister(self.ptr.as_ptr() as *mut c_void);
            }
        }

        #[cfg(target_os = "linux")]
        unsafe {
            let _ = libc::munmap(self.ptr.as_ptr() as *mut c_void, self.size);
        }

        #[cfg(not(target_os = "linux"))]
        unsafe {
            if let Ok(layout) = std::alloc::Layout::from_size_align(self.size, HUGEPAGE_SIZE) {
                std::alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}
