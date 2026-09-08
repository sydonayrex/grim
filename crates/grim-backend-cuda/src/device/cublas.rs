//! cuBLAS handle wrapper with safe thread serialization and RAII cleanup.

use crate::device::handles::cublasDestroy_v2;
use std::ffi::c_void;

/// Wrapper making cuBLAS FFI types Send + Sync.
/// # Safety `CublasHandle` wraps a `*mut c_void` cuBLAS handle obtained from `cublasCreate_v2`.
#[derive(Debug)]
pub struct CublasHandle(pub *mut c_void);

// SAFETY: `CublasHandle` wraps a raw CUDA driver handle.
// `Send` is safe because the handle is bound to a specific CUDA context on one.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid cuBLAS handle created via `cublasCreate_v2`.
        // It is destroyed exactly once, when the last `CudaDevice` clone sharing this `Arc<Mutex<Option<CublasHandle>>>` is dropped.
        if !self.0.is_null() {
            unsafe {
                let _ = cublasDestroy_v2(self.0);
            }
        }
    }
}
