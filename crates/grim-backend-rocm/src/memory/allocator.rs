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
    /// P0-1: each entry carries an event recorded on the stream that was
    /// current at free() time; a later `alloc` reusing the block waits on
    /// that event first, so an in-flight consumer on another stream can
    /// never race the new owner's writes.
    pool: Mutex<HashMap<usize, Vec<PoolEntry>>>,
    /// Total bytes currently held in `pool` (not returned to the driver).
    cached_bytes: Mutex<usize>,
    /// Soft cap on `cached_bytes`. Once exceeded, freed buffers are actually [see: `hipFree`]
    cap_bytes: usize,
    /// Device ordinal this allocator serves. Pinned around every driver call so a thread whose
    /// HIP context drifted to another device cannot `hipMalloc`/`hipFree` against the wrong one (WI-M1 context discipline).
    ordinal: usize,
    /// Count of real `hipMalloc` calls (misses). Always incremented.
    malloc_count: AtomicUsize,
    /// Count of real `hipFree` calls (evictions / cap overflow). Always incremented.
    free_count: AtomicUsize,
}

/// A pooled device block plus the fence event recorded when it was freed.
#[derive(Debug)]
struct PoolEntry {
    ptr: u64,
    event: *mut c_void,
}

unsafe impl Send for PoolEntry {}

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
        if let Some(entry) = reused {
            // Buffer leaves the pool: adjust cached accounting, and order the
            // new owner's stream after whatever last used this block.
            if let Ok(mut cached) = self.cached_bytes.lock() {
                *cached = cached.saturating_sub(cls);
            }
            if !entry.event.is_null() {
                let stream = crate::device::roc_device::RocmDevice::shared(self.ordinal)
                    .active_stream();
                if !stream.is_null() {
                    // SAFETY: event owned by this entry; stream is live.
                    unsafe {
                        let _ = crate::hipStreamWaitEvent(stream, entry.event, 0);
                    }
                }
            }
            return Ok(entry.ptr as *mut c_void);
        }

        // WI-M1: `hipMalloc` allocates in the calling thread's current device context.
        // Pin this allocator's ordinal - a pool miss from a drifted thread must not materialise.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut dev_ptr_void: *mut c_void = std::ptr::null_mut();
        let res = check_hip("hipMalloc", unsafe { hipMalloc(&mut dev_ptr_void, cls) });
        if res.is_err() {
            self.empty_cache();
            check_hip("hipMalloc", unsafe { hipMalloc(&mut dev_ptr_void, cls) })?;
        }
        drop(_guard);
        self.malloc_count.fetch_add(1, Ordering::Relaxed);
        Ok(dev_ptr_void)
    }

    /// Return a buffer to the pool (or actually free it if over cap).
    pub fn free(&self, ptr: *mut c_void, bytes: usize) {
        // TEMP-DIAG (GGUF fault hunt): GRIM_ALLOC_NO_POOL=1 makes every free a synchronized real release, ruling
        // pool reuse in/out as the cause of the "Page not present" GPU fault.
        if std::env::var("GRIM_ALLOC_NO_POOL").is_ok() {
            // WI-M1: pin the owning ordinal for the real release (see below).
            let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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
            // P0-1 (PLAN-improve-grim-perf): a real release must be ordered
            // after EVERY in-flight consumer, not just null-stream work. The
            // previous `hipFreeAsync(ptr, null stream)` could unmap a page
            // while a compute-stream kernel still read it — the "Page not
            // present" UAF (fused Q8_0 QKV fault, GPU 1 speed test). Eviction
            // is rare (only over cap), so a device-wide synchronize before
            // the free costs nothing measurable and is correct for any stream.
            let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
            unsafe {
                let _ = crate::hipDeviceSynchronize();
                let _ = hipFree(ptr);
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // P0-1: fence the block against in-flight consumers. Record an event
        // on the stream that was current at free() time — any future reuse
        // waits on it, so a queued kernel reading this buffer can never race
        // the new owner's writes, regardless of which stream either ran on.
        let fence = crate::device::roc_device::RocmDevice::shared(self.ordinal)
            .active_stream();
        let event: *mut c_void = if fence.is_null() {
            std::ptr::null_mut()
        } else {
            let mut ev: *mut c_void = std::ptr::null_mut();
            // SAFETY: fresh event handle; stream is the device's live stream.
            if unsafe { crate::hipEventCreate(&mut ev) } == 0 && !fence.is_null() {
                // SAFETY: as above.
                unsafe {
                    let _ = crate::hipEventRecord(ev, fence);
                }
                ev
            } else {
                std::ptr::null_mut()
            }
        };
        {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.entry(cls).or_default().push(PoolEntry { ptr: ptr as u64, event });
            let mut cached = self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner());
            *cached += cls;
        }
    }

    /// Release every pooled buffer back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        // Pin the device (P1-7 discipline): hipDeviceSynchronize targets the calling thread's current device, which may
        // differ from `self.ordinal` on a multi-GPU host where another device's teardown ran on this thread.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        unsafe {
            let _ = crate::hipDeviceSynchronize();
        }
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        for (_cls, bufs) in pool.drain() {
            for p in bufs {
                unsafe {
                    let _ = hipFree(p.ptr as *mut c_void);
                    if !p.event.is_null() {
                        let _ = crate::hipEventDestroy(p.event);
                    }
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
