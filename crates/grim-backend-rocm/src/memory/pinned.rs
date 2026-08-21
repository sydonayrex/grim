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
