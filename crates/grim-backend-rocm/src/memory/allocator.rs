//! Implementation of `RocmCachingAllocator` — a size-bucketed free-list [see: `hipMalloc`, `hipFree`, `RocmStorage::drop`, `Send + Sync`]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grim_tensor::error::Result;

use crate::{check_hip, hipFree, hipMalloc};

/// Size-bucketed caching allocator for device memory. [see: `hipMalloc`, `hipFree`, `Drop for RocmStorage`, `Arc`]
#[derive(Debug)]
pub struct RocmCachingAllocator {
    /// Free-list: size class -> available device pointers (stored as `u64` so the [see: `Send + Sync`]
    pool: Mutex<HashMap<usize, Vec<u64>>>,
    /// Total bytes currently held in `pool` (not returned to the driver).
    cached_bytes: Mutex<usize>,
    /// Soft cap on `cached_bytes`. Once exceeded, freed buffers are actually [see: `hipFree`]
    cap_bytes: usize,
    /// Device ordinal this allocator serves.
    #[allow(dead_code)]
    ordinal: usize,
    /// Count of real `hipMalloc` calls (misses). Always incremented.
    malloc_count: AtomicUsize,
    /// Count of real `hipFree` calls (evictions / cap overflow). Always incremented.
    free_count: AtomicUsize,
}

impl RocmCachingAllocator {
    pub fn new(ordinal: usize, cap_bytes: usize) -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
            cached_bytes: Mutex::new(0),
            cap_bytes,
            ordinal,
            malloc_count: AtomicUsize::new(0),
            free_count: AtomicUsize::new(0),
        }
    }

    /// Round a byte size up to the next power of two. Class 0 is treated as 1 to [see: `hipMalloc`]
    fn size_class(bytes: usize) -> usize {
        if bytes <= 1 {
            1
        } else {
            bytes.next_power_of_two()
        }
    }

    /// Allocate a device buffer of at least `bytes` usable bytes, reusing a pooled
    pub fn alloc(&self, bytes: usize) -> Result<*mut c_void> {
        let cls = Self::size_class(bytes);
        let reused = {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.get_mut(&cls).and_then(|v| v.pop())
        };
        if let Some(ptr_u64) = reused {
            // Buffer leaves the pool: adjust cached accounting.
            if let Ok(mut cached) = self.cached_bytes.lock() {
                *cached = cached.saturating_sub(cls);
            }
            return Ok(ptr_u64 as *mut c_void);
        }

        let mut dev_ptr_void: *mut c_void = std::ptr::null_mut();
        let res = check_hip("hipMalloc", unsafe { hipMalloc(&mut dev_ptr_void, cls) });
        if res.is_err() {
            self.empty_cache();
            check_hip("hipMalloc", unsafe { hipMalloc(&mut dev_ptr_void, cls) })?;
        }
        self.malloc_count.fetch_add(1, Ordering::Relaxed);
        Ok(dev_ptr_void)
    }

    /// Return a buffer to the pool (or actually free it if over cap).
    pub fn free(&self, ptr: *mut c_void, bytes: usize) {
        // TEMP-DIAG (GGUF fault hunt): GRIM_ALLOC_NO_POOL=1 makes every free
        // a synchronized real release, ruling pool reuse in/out as the cause
        // of the "Page not present" GPU fault.
        if std::env::var("GRIM_ALLOC_NO_POOL").is_ok() {
            unsafe {
                let _ = crate::hipDeviceSynchronize();
                let _ = hipFree(ptr);
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let cls = Self::size_class(bytes);
        let over_cap = {
            let cached = self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner());
            *cached + cls > self.cap_bytes
        };
        if over_cap || ptr.is_null() {
            // Use hipFreeAsync on the null stream instead of hipDeviceSynchronize + hipFree.
            // hipFreeAsync enqueues the release after all currently submitted work on the
            // null stream, avoiding a full-device stall. Falls back to sync hipFree if the
            // async path is unavailable (pre-ROCm-5.4 drivers return an error code).
            // CONTRACT: ptr must not be reused by the caller after this call returns.
            unsafe {
                let res = crate::hipFreeAsync(ptr, std::ptr::null_mut());
                if res != 0 {
                    // hipFreeAsync not supported — fall back to sync path.
                    let _ = crate::hipDeviceSynchronize();
                    let _ = hipFree(ptr);
                }
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.entry(cls).or_default().push(ptr as u64);
            let mut cached = self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner());
            *cached += cls;
        }
    }

    /// Release every pooled buffer back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        // Pin the device (P1-7 discipline): hipDeviceSynchronize targets the
        // calling thread's current device, which may differ from `self.ordinal`
        // on a multi-GPU host where another device's teardown ran on this thread.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        unsafe {
            let _ = crate::hipDeviceSynchronize();
        }
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        for (_cls, bufs) in pool.drain() {
            for p in bufs {
                unsafe {
                    let _ = hipFree(p as *mut c_void);
                }
                self.free_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        *self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner()) = 0;
    }

    /// `(malloc_count, free_count)` — real driver allocation calls since start.
    pub fn stats(&self) -> (usize, usize) {
        (
            self.malloc_count.load(Ordering::Relaxed),
            self.free_count.load(Ordering::Relaxed),
        )
    }
}
