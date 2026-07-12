//! Pinned (`hipHostMalloc`) page-locked host buffer used by the per-token
//! decode hot path. Pinned memory transfers over PCIe/xGMI at full bandwidth
//! with `hipMemcpyAsync`, whereas pageable `Vec` staging forces a slower
//! bounce buffer.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 9 — pinned staging for repeated
//!   small host→device / device→host transfers in the decode loop.
//! - `rocm-profiling-perf` — host transfer is in the per-call hot path;
//!   pre-pinning amortizes the registration cost across all subsequent calls.

use std::ffi::c_void;
use std::marker::PhantomData;

use grim_tensor::error::{Error, Result};

use crate::{hipHostFree, hipHostMalloc, HipErrorT, hipSuccess};

/// A host-side staging buffer allocated with `hipHostMalloc` (pinned / page-locked
/// memory). Pinned buffers transfer over PCIe/xGMI at full bandwidth with
/// `hipMemcpyAsync`, whereas pageable `Vec` staging forces a slower bounce buffer.
///
/// This is the building block for the per-token decode hot path (feeding a sampled
/// token in, reading logits out): the caller keeps one `RocmPinnedBuffer` per
/// recurring transfer and reuses it across steps instead of allocating fresh each
/// time. Cold-path / one-off transfers continue to use plain `Vec` + synchronous
/// `hipMemcpy`.
pub struct RocmPinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// The buffer is only touched from the owning thread; the raw pointer is not shared.
unsafe impl<T: Send> Send for RocmPinnedBuffer<T> {}

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
        let res: HipErrorT = unsafe { hipHostMalloc(&mut ptr, len * std::mem::size_of::<T>(), 0) };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipHostMalloc failed with error code {}",
                res
            )));
        }
        Ok(RocmPinnedBuffer {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Allocate pinned memory and copy `slice` into it.
    pub fn from_slice(slice: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(slice.len())?;
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
