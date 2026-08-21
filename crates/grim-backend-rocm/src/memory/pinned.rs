//! Pinned (`hipHostMalloc`) page-locked host buffer used by the per-token [see: `hipMemcpyAsync`, `Vec`]

use std::ffi::c_void;
use std::marker::PhantomData;

use grim_tensor::error::Result;

use crate::{check_hip, hipHostFree, hipHostMalloc};

/// A host-side staging buffer allocated with `hipHostMalloc` (pinned / page-locked [see: `hipMemcpyAsync`, `Vec`]
pub struct RocmPinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// The buffer is only touched from the owning thread; the raw pointer is not shared.
unsafe impl<T: Send> Send for RocmPinnedBuffer<T> {}

impl<T> std::fmt::Debug for RocmPinnedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmPinnedBuffer")
            .field("len", &self.len)
            .finish()
    }
}

impl<T: Copy> RocmPinnedBuffer<T> {
    /// Allocate `len` elements of pinned host memory.
    pub fn alloc(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(RocmPinnedBuffer {
                ptr: std::ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            });
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // flags = 0 → default portable pinned memory (hipHostMallocDefault).
        check_hip("hipHostMalloc", unsafe {
            hipHostMalloc(&mut ptr, len * std::mem::size_of::<T>(), 0)
        })?;
        Ok(RocmPinnedBuffer {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Allocate pinned memory and copy `slice` into it.
    pub fn from_slice(slice: &[T]) -> Result<Self> {
        let buf = Self::alloc(slice.len())?;
        if !slice.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), buf.ptr, slice.len());
            }
        }
        Ok(buf)
    }

    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for RocmPinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = hipHostFree(self.ptr as *mut c_void);
            }
        }
    }
}

/// A thread-safe pool of reusable pinned host buffers for high-speed PCIe DMA streaming.
pub struct PinnedStagingPool {
    capacity_elements: usize,
    free_buffers: std::sync::Mutex<Vec<RocmPinnedBuffer<u8>>>,
}

impl PinnedStagingPool {
    /// Create a new pinned staging pool with pre-allocated buffer capacity.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_elements: capacity_bytes,
            free_buffers: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Acquire a pinned host buffer of at least `min_bytes` capacity.
    pub fn acquire(&self, min_bytes: usize) -> Result<RocmPinnedBuffer<u8>> {
        let mut lock = self.free_buffers.lock().unwrap();
        if let Some(pos) = lock.iter().position(|b| b.len() >= min_bytes) {
            Ok(lock.swap_remove(pos))
        } else {
            let alloc_size = min_bytes.max(self.capacity_elements);
            RocmPinnedBuffer::alloc(alloc_size)
        }
    }

    /// Release a buffer back into the pool for reuse.
    pub fn release(&self, buffer: RocmPinnedBuffer<u8>) {
        let mut lock = self.free_buffers.lock().unwrap();
        if lock.len() < 8 {
            lock.push(buffer);
        }
    }
}
