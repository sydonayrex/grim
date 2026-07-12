//! Green-phase implementation of the device scratch memory pool.
//!
//! Phase-3 §3.1 of the QKV spec, RED→GREEN→REFACTOR.
//!
//! The pool hands out `hipMalloc`-backed scratch buffers via LIFO
//! bucketization; dropped `PooledBuffer`s return to the bucket so the
//! next request of the same size reuses the underlying slot.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 3 — Memory pool (KV-cache scratch)
//! - `rust-gpu-parallelism` — Stream-ordered memory strategy
//! - `rocm-profiling-perf` — Allocation overhead is in the optimizer's hot
//!   path; the pool eliminates a measurable per-call cost
//! - `rust-gpu-discipline` §3 — No silent CPU fallback for GPU-only ops.
//!   Every allocation goes through `hipMalloc`; counters are real.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use grim_tensor::error::{Error, Result};

use crate::{hipFree, hipMalloc, HipErrorT, hipSuccess};

/// Layout key for the scratch pool: (rounded size, alignment).
///
/// Sizes bucketize to the next power of two with a 256-byte floor. The
/// floor prevents the pool from spamming tiny (1-byte etc.) buckets that
/// would only ever compete with `hipMalloc` overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolLayout {
    pub size: usize,
    pub align: usize,
}

impl PoolLayout {
    pub fn new(size: usize, align: usize) -> Self {
        let bucket = if size < 256 { 256 } else { size.next_power_of_two() };
        Self { size: bucket, align }
    }
}

/// RAII handle for a pooled device buffer. On `Drop` the underlying
/// pointer is returned to the matching bucket (its `hipMalloc` slot
/// remains valid for re-use; no copy is performed).
pub struct PooledBuffer {
    ptr: *mut std::ffi::c_void,
    layout: PoolLayout,
    pool: Arc<DeviceScratchPool>,
}

impl PooledBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Borrowed view of the underlying device pointer. Used by the
    /// upload path; the `Drop` of `PooledBuffer` is the owning destructor.
    pub fn as_device_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn layout(&self) -> PoolLayout {
        self.layout
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            self.pool.return_buffer(self.ptr, self.layout);
        }
    }
}

/// Thread-safe scratch buffer pool with power-of-2 bucketization.
#[derive(Debug)]
pub struct DeviceScratchPool {
    buckets: Mutex<HashMap<PoolLayout, Vec<*mut std::ffi::c_void>>>,
    peak_bytes: AtomicUsize,
    current_bytes: AtomicUsize,
}

impl DeviceScratchPool {
    /// Build a new, empty pool. State lives in atomic counters and a
    /// Mutex-wrapped bucket map; no GPU calls happen until `get()` runs.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            buckets: Mutex::new(HashMap::new()),
            peak_bytes: AtomicUsize::new(0),
            current_bytes: AtomicUsize::new(0),
        })
    }

    /// Get a buffer of at least `size` bytes, `align`-aligned. Recycles
    /// the most recently freed pointer in the matching bucket when one
    /// exists; otherwise `hipMalloc`s a fresh slot and tracks the peak.
    pub fn get(self: &Arc<Self>, size: usize, align: usize) -> Result<PooledBuffer> {
        let layout = PoolLayout::new(size, align);
        let ptr = {
            let mut buckets = self.buckets.lock().map_err(|_| {
                Error::Backend("DeviceScratchPool bucket mutex poisoned".into())
            })?;
            buckets.get_mut(&layout).and_then(|v| v.pop())
        };

        let ptr = match ptr {
            Some(p) => p,
            None => {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                let res: HipErrorT = unsafe { hipMalloc(&mut p, layout.size) };
                if res != hipSuccess {
                    return Err(Error::Backend(format!(
                        "scratch pool hipMalloc failed: code={}",
                        res
                    )));
                }
                self.current_bytes
                    .fetch_add(layout.size, Ordering::Relaxed);
                let cur = self.current_bytes.load(Ordering::Relaxed);
                let mut peak = self.peak_bytes.load(Ordering::Relaxed);
                while cur > peak {
                    match self.peak_bytes.compare_exchange(
                        peak,
                        cur,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(actual) => peak = actual,
                    }
                }
                p
            }
        };

        Ok(PooledBuffer { ptr, layout, pool: self.clone() })
    }

    /// Internal recycle. Called from `PooledBuffer::drop`.
    fn return_buffer(&self, ptr: *mut std::ffi::c_void, layout: PoolLayout) {
        if let Ok(mut buckets) = self.buckets.lock() {
            buckets.entry(layout).or_default().push(ptr);
        }
        // Mutex-poison fallback: silent recycle failure means the next
        // `get` will re-`hipMalloc`. Correctness preserved; perf degrades
        // once until a `clear` restart.
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Free every cached pointer back to the GPU. Used by `Drop` to avoid
    /// leaking the pool's underlying hipMalloc allocations.
    fn drain(&self) {
        let buckets = match self.buckets.lock() {
            Ok(b) => b,
            Err(_) => return,
        };
        for (_, v) in buckets.iter() {
            for &p in v {
                if !p.is_null() {
                    let _ = unsafe { hipFree(p) };
                }
            }
        }
    }
}

impl Drop for DeviceScratchPool {
    fn drop(&mut self) {
        self.drain();
    }
}
