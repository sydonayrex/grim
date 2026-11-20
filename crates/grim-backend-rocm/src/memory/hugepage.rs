//! 2MB Linux HugePage Host-Pinned Memory Allocator.
//! Provides `HugePagePinnedBuffer`, which allocates 2MB-aligned host memory via `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` on Linux (falling back.

use crate::device::handles::{hipHostRegister, hipHostUnregister};
use grim_tensor::error::{Error, Result};
use std::ffi::c_void;
use std::ptr::NonNull;

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

        // Register host memory with HIP for zero-copy DMA access unless UMA unpinned guard is active.
        let skip_pin = std::env::var("GRIM_UMA_UNPINNED").ok().as_deref() == Some("1")
            && should_skip_uma_host_pin(
                std::env::var("GRIM_GFX_ARCH").ok().as_deref().unwrap_or(""),
                true,
            );
        let is_registered = if skip_pin {
            false
        } else {
            let res = unsafe { hipHostRegister(ptr.as_ptr() as *mut c_void, size, 0) };
            res == 0
        };

        Ok(Self {
            ptr,
            size,
            is_hugepage,
            is_registered,
        })
    }

    #[cfg(target_os = "linux")]
    fn alloc_mmap(size: usize) -> Result<(NonNull<u8>, bool)> {
        let try_huge = if std::env::var("GRIM_THP_GUARD").ok().as_deref() == Some("1") {
            let thp = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
                .unwrap_or_default();
            let buddy = std::fs::read_to_string("/proc/buddyinfo").unwrap_or_default();
            should_madvise_hugepage_from_proc(&thp, &buddy, size)
        } else {
            true
        };
        if try_huge {
            // Try 2MB Hugepages first
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

/// Parse `/sys/kernel/mm/transparent_hugepage/enabled` and `/proc/buddyinfo`.
/// Returns `true` iff THP is NOT `[always]` and order-9/10 free blocks cover >=50% of `bytes`.
pub fn should_madvise_hugepage_from_proc(
    thp_enabled: &str,
    buddyinfo: &str,
    bytes: usize,
) -> bool {
    if thp_enabled.contains("[always]") {
        return false;
    }
    let needed_2m_pages = bytes.div_ceil(HUGEPAGE_SIZE);
    if needed_2m_pages == 0 {
        return false;
    }
    let mut free_2m_equiv = 0usize;
    for line in buddyinfo.lines() {
        let Some(idx) = line.find("Normal") else {
            continue;
        };
        let orders: Vec<usize> = line[idx + "Normal".len()..]
            .split_whitespace()
            .filter_map(|s| s.parse::<usize>().ok())
            .collect();
        if orders.len() >= 11 {
            free_2m_equiv += orders[9] + 2 * orders[10];
        }
    }
    free_2m_equiv * 2 >= needed_2m_pages
}

/// Returns `true` if `hipHostRegister` must be skipped on UMA APUs (`gfx1103`, `gfx1151` Strix Halo).
pub fn should_skip_uma_host_pin(gfx_arch: &str, is_integrated_apu: bool) -> bool {
    is_integrated_apu || gfx_arch.contains("gfx1103") || gfx_arch.contains("gfx1151")
}

/// Parse cgroup v2 `memory.stat` reclaimable file cache (`conversation_memory.cpp:91-185`).
pub fn compute_cgroup_v2_reclaimable_bytes(memory_stat: &str) -> u64 {
    let mut inactive_file = 0u64;
    let mut file_dirty = 0u64;
    let mut file_writeback = 0u64;
    for line in memory_stat.lines() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next().and_then(|v| v.parse::<u64>().ok())) {
            (Some("inactive_file"), Some(v)) => inactive_file = v,
            (Some("file_dirty"), Some(v)) => file_dirty = v,
            (Some("file_writeback"), Some(v)) => file_writeback = v,
            _ => {}
        }
    }
    inactive_file.saturating_sub(file_dirty.saturating_add(file_writeback))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPinBudget {
    pub soft_reserve_bytes: u64,
    pub hard_reserve_bytes: u64,
    pub uma_reserve_bytes: u64,
    pub soft_budget_bytes: u64,
    pub hard_budget_bytes: u64,
}

/// Two-tier Soft/Hard/UMA host-memory pin budget.
pub fn compute_host_pin_budget(
    mem_total: u64,
    mem_avail: u64,
    cgroup_reclaimable: Option<u64>,
    is_uma: bool,
) -> HostPinBudget {
    const GIB: u64 = 1024 * 1024 * 1024;
    let effective_avail = cgroup_reclaimable.map_or(mem_avail, |cg| mem_avail.min(cg));
    let soft_reserve = (8 * GIB).max(mem_total / 5);
    let hard_reserve = (4 * GIB).max(mem_total / 10);
    let uma_reserve = if is_uma {
        (8 * GIB).max(mem_total / 12)
    } else {
        0
    };
    let active_soft_res = if is_uma { uma_reserve } else { soft_reserve };
    HostPinBudget {
        soft_reserve_bytes: soft_reserve,
        hard_reserve_bytes: hard_reserve,
        uma_reserve_bytes: uma_reserve,
        soft_budget_bytes: effective_avail.saturating_sub(active_soft_res),
        hard_budget_bytes: effective_avail.saturating_sub(hard_reserve),
    }
}
