//! Green-phase implementation of the device scratch memory pool. [see: `hipMalloc`, `PooledBuffer`]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use grim_tensor::error::{Error, Result};

use crate::{check_hip, hipFree, hipMalloc};

/// Layout key for the scratch pool: (rounded size, alignment). [see: `hipMalloc`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolLayout {
    pub size: usize,
    pub align: usize,
}

impl PoolLayout {
    pub fn new(size: usize, align: usize) -> Self {
        let bucket = if size < 256 {
            256
        } else {
            size.next_power_of_two()
        };
        Self {
            size: bucket,
            align,
        }
    }
}

/// RAII handle for a pooled device buffer. On `Drop` the underlying [see: `hipMalloc`]
pub struct PooledBuffer {
    ptr: *mut std::ffi::c_void,
    layout: PoolLayout,
    pool: Arc<DeviceScratchPool>,
}

impl PooledBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Borrowed view of the underlying device pointer. Used by the [see: `Drop`, `PooledBuffer`]
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
    /// Build a new, empty pool. State lives in atomic counters and a [see: `get()`]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            buckets: Mutex::new(HashMap::new()),
            peak_bytes: AtomicUsize::new(0),
            current_bytes: AtomicUsize::new(0),
        })
    }

    /// Get a buffer of at least `size` bytes, `align`-aligned. Recycles [see: `hipMalloc`]
    pub fn get(self: &Arc<Self>, size: usize, align: usize) -> Result<PooledBuffer> {
        let layout = PoolLayout::new(size, align);
        let ptr = {
            let mut buckets = self
                .buckets
                .lock()
                .map_err(|_| Error::Backend("DeviceScratchPool bucket mutex poisoned".into()))?;
            buckets.get_mut(&layout).and_then(|v| v.pop())
        };

        let ptr = match ptr {
            Some(p) => p,
            None => {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                check_hip("scratch pool hipMalloc", unsafe {
                    hipMalloc(&mut p, layout.size)
                })?;
                self.current_bytes.fetch_add(layout.size, Ordering::Relaxed);
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

        Ok(PooledBuffer {
            ptr,
            layout,
            pool: self.clone(),
        })
    }

    /// Internal recycle. Called from `PooledBuffer::drop`.
    fn return_buffer(&self, ptr: *mut std::ffi::c_void, layout: PoolLayout) {
        if let Ok(mut buckets) = self.buckets.lock() {
            buckets.entry(layout).or_default().push(ptr);
        }
        // Mutex-poison fallback: silent recycle failure means the next [see: `get`, `hipMalloc`]
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Free every cached pointer back to the GPU. Used by `Drop` to avoid
    fn drain(&self) {
        let mut buckets = match self.buckets.lock() {
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
        // MOD-1 fix: clear the bucket map so current_bytes() reflects reality.
        buckets.clear();
        self.current_bytes.store(0, Ordering::Relaxed);
    }
}

impl Drop for DeviceScratchPool {
    fn drop(&mut self) {
        self.drain();
    }
}
