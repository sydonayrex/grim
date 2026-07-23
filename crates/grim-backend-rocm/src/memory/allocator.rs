//! Implementation of `RocmCachingAllocator` — a size-bucketed free-list
//! allocator that avoids `hipMalloc`/`hipFree` round-trips on the decode
//! hot path.
//!
//! Items 1 and 7 of the ROCm spec: power-of-two bucketization with a byte-cap,
//! per-device state, full-drop semantics on `RocmStorage::drop`. The struct
//! keeps `Send + Sync` by storing device pointers as `u64` (numeric identity
//! is sufficient for the free-list — we don't dereference the pointer on the
//! CPU side, only pass it back to `hipFree` later).
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 3 — Memory pool (the alloc hot path)
//! - `rocm-profiling-perf` — measurable per-call `hipMalloc` cost is what
//!   motivated the pool
//! - `rust-gpu-discipline` §3 — never silently fall back to a CPU path; on
//!   allocation failure we return `Err` so the caller can decide.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use grim_tensor::error::Result;

use crate::{check_hip, hipFree, hipMalloc};

/// Size-bucketed caching allocator for device memory.
///
/// `hipMalloc`/`hipFree` are effectively device-synchronizing driver calls, so
/// calling them per-op in a decode loop is the dominant fixed per-token cost.
/// This allocator keeps a free-list of device buffers keyed by power-of-two size
/// class per device and reuses them across allocations, only falling back to the
/// real `hipMalloc`/`hipFree` when the pool is empty or over its soft cap.
///
/// Buffers are returned to the pool by `Drop for RocmStorage` (which holds an
/// `Arc` to this allocator), so callers do not need to manage reuse explicitly.
#[derive(Debug)]
pub struct RocmCachingAllocator {
    /// Free-list: size class -> available device pointers (stored as `u64` so the
    /// pool stays `Send + Sync`; device pointers are just integers).
    pool: Mutex<HashMap<usize, Vec<u64>>>,
    /// Total bytes currently held in `pool` (not returned to the driver).
    cached_bytes: Mutex<usize>,
    /// Soft cap on `cached_bytes`. Once exceeded, freed buffers are actually
    /// `hipFree`'d instead of retained, bounding steady-state memory use.
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

    /// Round a byte size up to the next power of two. Class 0 is treated as 1 to
    /// avoid a zero-sized `hipMalloc`.
    fn size_class(bytes: usize) -> usize {
        if bytes <= 1 {
            1
        } else {
            bytes.next_power_of_two()
        }
    }

    /// Allocate a device buffer of at least `bytes` usable bytes, reusing a pooled
    /// buffer when one is available.
    pub fn alloc(&self, bytes: usize) -> Result<*mut c_void> {
        let cls = Self::size_class(bytes);
        let reused = {
            let mut pool = self.pool.lock().unwrap();
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
        let cls = Self::size_class(bytes);
        let over_cap = {
            let cached = self.cached_bytes.lock().unwrap();
            *cached + cls > self.cap_bytes
        };
        if over_cap || ptr.is_null() {
            unsafe {
                let _ = crate::hipDeviceSynchronize();
                let _ = hipFree(ptr);
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        {
            let mut pool = self.pool.lock().unwrap();
            pool.entry(cls).or_default().push(ptr as u64);
            let mut cached = self.cached_bytes.lock().unwrap();
            *cached += cls;
        }
    }

    /// Release every pooled buffer back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        unsafe {
            let _ = crate::hipDeviceSynchronize();
        }
        let mut pool = self.pool.lock().unwrap();
        for (_cls, bufs) in pool.drain() {
            for p in bufs {
                unsafe {
                    let _ = hipFree(p as *mut c_void);
                }
                self.free_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        *self.cached_bytes.lock().unwrap() = 0;
    }

    /// `(malloc_count, free_count)` — real driver allocation calls since start.
    pub fn stats(&self) -> (usize, usize) {
        (
            self.malloc_count.load(Ordering::Relaxed),
            self.free_count.load(Ordering::Relaxed),
        )
    }
}
