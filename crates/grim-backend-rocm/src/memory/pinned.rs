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
    /// Allocate `len` elements of pinned host memory with the thread's HIP
    /// context pinned to `ordinal` — scythe2/GRAVE lesson: unpinned
    /// `hipHostMalloc` can land on a foreign device context and page-fault on
    /// later copies. New code must use this, not `alloc`.
    pub fn alloc_on(ordinal: usize, len: usize) -> Result<Self> {
        let _guard = crate::device::util::DeviceGuard::set(ordinal as i32);
        Self::alloc(len)
    }

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
        let mut lock = self.free_buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = lock.iter().position(|b| b.len() >= min_bytes) {
            Ok(lock.swap_remove(pos))
        } else {
            let alloc_size = min_bytes.max(self.capacity_elements);
            RocmPinnedBuffer::alloc(alloc_size)
        }
    }

    /// Release a buffer back into the pool for reuse.
    pub fn release(&self, buffer: RocmPinnedBuffer<u8>) {
        let mut lock = self.free_buffers.lock().unwrap_or_else(|e| e.into_inner());
        if lock.len() < 8 {
            lock.push(buffer);
        }
    }
}

thread_local! {
    static H2D_STAGE_RING: std::cell::RefCell<Vec<(usize, StageRing)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

const STAGE_RING_SLOTS: usize = 16;
const STAGE_RING_MAX_BYTES: usize = 64 * 1024;

struct StageRing {
    slots: Vec<RocmPinnedBuffer<u8>>,
    next: usize,
}

/// Stage `host` into the ring slot for `ordinal` and return the pinned pointer.
/// Falls back to the input pointer for oversized uploads (caller then copies
/// from pageable memory, as before).
// Thread-local pinned staging ring for small per-step H2D uploads (token ids,
// positions, scalars). Pageable-memory `hipMemcpyAsync` takes the driver's
// slow staged path (~0.4 ms per call observed on gfx1201), which dominated
// decode time; copying into pinned memory first avoids it entirely.
// SAFETY: the returned pointer stays valid until this thread stages
// `SLOTS` more times. Decode steps end with a blocking sampled-token D2H
// (stream fully drained), and a step issues far fewer than `SLOTS` staged
// copies, so a slot is never rewritten while its copy is in flight.
pub fn stage_h2d_pinned<'a>(ordinal: usize, host: &'a [u8]) -> (*const u8, bool) {
    if host.is_empty() || host.len() > STAGE_RING_MAX_BYTES {
        return (host.as_ptr(), false);
    }
    H2D_STAGE_RING.with(|cell| {
        let mut rings = cell.borrow_mut();
        let entry = match rings.iter_mut().find(|(o, _)| *o == ordinal) {
            Some(e) => &mut e.1,
            None => {
                rings.push((
                    ordinal,
                    StageRing {
                        slots: Vec::new(),
                        next: 0,
                    },
                ));
                &mut rings.last_mut().unwrap().1
            }
        };
        if entry.slots.len() < STAGE_RING_SLOTS {
            // scythe2 lesson: hipHostMalloc is a raw HIP seam — pin the
            // thread's context to the owning ordinal or the pinned pages can
            // land on a foreign device context (page fault on later copies).
            let _guard = crate::device::util::DeviceGuard::set(ordinal as i32);
            match RocmPinnedBuffer::<u8>::alloc(STAGE_RING_MAX_BYTES) {
                Ok(buf) => entry.slots.push(buf),
                Err(_) => return (host.as_ptr(), false),
            }
        }
        let idx = entry.next % entry.slots.len();
        entry.next = entry.next.wrapping_add(1);
        let slot = &mut entry.slots[idx];
        let dst = slot.as_mut_slice();
        dst[..host.len()].copy_from_slice(host);
        (dst.as_ptr(), true)
    })
}
