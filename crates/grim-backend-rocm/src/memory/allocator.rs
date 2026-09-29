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
    /// Unique instance id (audit diagnosis: instance address reuse lies).
    id: usize,
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
    /// GRIM_ALLOC_AUDIT=1: live-set integrity audit. Maps every handed-out
    /// pointer to its size class; a pointer handed out twice while live, or
    /// freed twice, is the deterministic aliasing that produced the d2d
    /// decode gibberish.
    live: Mutex<std::collections::HashMap<u64, usize>>,
}

/// A pooled device block plus the fence event recorded when it was freed.
#[derive(Debug)]
struct PoolEntry {
    ptr: u64,
    event: *mut c_void,
}

unsafe impl Send for PoolEntry {}

static NEXT_ALLOCATOR_ID: AtomicUsize = AtomicUsize::new(1);

impl RocmCachingAllocator {
    pub fn new(ordinal: usize, cap_bytes: usize) -> Self {
        let id = NEXT_ALLOCATOR_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id,

            pool: Mutex::new(HashMap::new()),
            cached_bytes: Mutex::new(0),
            cap_bytes,
            ordinal,
            malloc_count: AtomicUsize::new(0),
            free_count: AtomicUsize::new(0),
            live: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Round a byte size up to the next power of two. Class 0 is treated as 1 to [see: `hipMalloc`]
    pub fn size_class_of(bytes: usize) -> usize { Self::size_class(bytes) }

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
            // new owner's writes after EVERY in-flight consumer of this block.
            //
            // The recorded event only fences the stream that was current at
            // `free()` time. The d2d decode path runs kernels on capture
            // streams that are not current when their output buffers are
            // dropped, so a stream-ordered wait let the new owner overwrite a
            // block a consumer on another stream was still reading/writing —
            // the Qwen3.5-9B d2d decode gibberish (h_normed [1024..4096)
            // garbage, GRIM_ALLOC_NO_POOL=1 clean). A device-wide
            // synchronize is correct for any stream; at decode's ~600
            // ms/token it is unmeasurable.
            self.audit_trace("POP ", entry.ptr, cls);
            self.audit_insert(entry.ptr, cls);
            if let Ok(mut cached) = self.cached_bytes.lock() {
                *cached = cached.saturating_sub(cls);
            }
            {
                let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
                unsafe {
                    let _ = crate::hipDeviceSynchronize();
                }
            }
            if !entry.event.is_null() {
                // The event fence is superseded by the synchronize above.
                unsafe {
                    let _ = crate::hipEventDestroy(entry.event);
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
        self.audit_trace("MALLOC", dev_ptr_void as u64, cls);
        self.audit_insert(dev_ptr_void as u64, cls);
        Ok(dev_ptr_void)
    }

    fn audit_on(&self) -> bool {
        std::env::var("GRIM_ALLOC_AUDIT").as_deref() == Ok("1")
    }

    fn audit_trace(&self, what: &str, ptr: u64, cls: usize) {
        if !self.audit_on() || cls != 16384 { return; }
        eprintln!("[alloc-trace] {} {ptr:#x} c{cls} id{}", what, self.id);
    }

    fn audit_insert(&self, ptr: u64, cls: usize) {
        if !self.audit_on() { return; }
        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = live.insert(ptr, cls) {
            eprintln!("[alloc-audit] DOUBLE-BOOKING: {ptr:#x} id{} handed out while live (class {old}, now {cls})", self.id);
        }
    }

    fn audit_remove(&self, ptr: u64, cls: usize) {
        if !self.audit_on() { return; }
        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        match live.remove(&ptr) {
            None => {
                eprintln!("[alloc-audit] FREE OF UNKNOWN: {ptr:#x} class {cls} not live");
                // One-shot backtrace names the duplicate-free caller.
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("{:?}", std::backtrace::Backtrace::force_capture());
                }
            }
            Some(c) if c != cls => eprintln!("[alloc-audit] CLASS MISMATCH on free: {ptr:#x} live {c}, freed as {cls} (id{}))", self.id),
            _ => {}
        }
    }

    /// Return a buffer to the pool (or actually free it if over cap).
    ///
    /// DEFAULT IS NO POOL. The size-classed free list has a broken ownership
    /// invariant: a block can be pooled while a duplicate owner still holds
    /// it, and the driver can then hand the same address to a fresh
    /// `hipMalloc` — two live owners, deterministic memory corruption. This
    /// was the Qwen3.5-9B d2d decode gibberish (arbitration: pool on ->
    /// h_normed[1024..4096) garbage and step-1 logprobs "&lt" -1.93; pool
    /// off -> byte-equal to the host reference). GRIM_ALLOC_AUDIT=1 traces
    /// the violations ([alloc-audit]/[alloc-trace]); GRIM_ALLOC_POOL=1 opts
    /// back into reuse for perf work once the duplicate-owner site is fixed.
    pub fn free(&self, ptr: *mut c_void, bytes: usize) {
        if std::env::var("GRIM_ALLOC_POOL").as_deref() != Ok("1") {
            // Correct-by-default path: device-wide synchronize (any stream's
            // in-flight consumer is done) then a real driver release.
            // WI-M1: pin the owning ordinal for the real release.
            let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
            unsafe {
                let _ = crate::hipDeviceSynchronize();
                let _ = hipFree(ptr);
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let cls = Self::size_class(bytes);
        self.audit_trace("FREE ", ptr as u64, cls);
        self.audit_remove(ptr as u64, cls);
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
        let fence = crate::device::roc_device::RocmDevice::shared(self.ordinal).active_stream();
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
            pool.entry(cls).or_default().push(PoolEntry {
                ptr: ptr as u64,
                event,
            });
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
