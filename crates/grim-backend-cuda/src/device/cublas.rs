//! cuBLAS handle wrapper with safe thread serialization and RAII cleanup.

use std::ffi::c_void;
use crate::device::handles::cublasDestroy_v2;

/// Wrapper making cuBLAS FFI types Send + Sync.
///
/// # Safety
/// `CublasHandle` wraps a `*mut c_void` cuBLAS handle obtained from
/// `cublasCreate_v2`. The handle is protected by an `Arc<Mutex<>>` in
/// `CudaDevice`, so concurrent access from multiple threads is
/// serialized. cuBLAS handles are thread-local in the driver; moving
/// the handle between threads is safe because the underlying driver
/// state is not thread-local — only concurrent *use* requires
/// synchronization, which the `Mutex` provides.
#[derive(Debug)]
pub struct CublasHandle(pub *mut c_void);

// SAFETY: `CublasHandle` wraps a raw CUDA driver handle. `Send` is safe because
// the handle is bound to a specific CUDA context on one device, and the driver
// tracks it independently of the creating thread. `Sync` is safe because the
// cuBLAS API serializes concurrent calls through its internal stream/context lock.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid cuBLAS handle created via `cublasCreate_v2`.
        // It is destroyed exactly once, when the last `CudaDevice` clone sharing
        // this `Arc<Mutex<Option<CublasHandle>>>` is dropped.
        if !self.0.is_null() {
            unsafe {
                let _ = cublasDestroy_v2(self.0);
            }
        }
    }
}
