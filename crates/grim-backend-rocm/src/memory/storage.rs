//! `RocmStorage`: ROCm-side device buffer + metadata, plus its
//! `BackendStorage` trait impl that integrates with the broader
//! `grim-tensor` backend abstraction.
//!
//! Allocation helpers (`RocmStorage::alloc_gpu`, `RocmStorage::copy_from_host`)
//! are kept in `lib.rs` as free-standing items because they need the full
//! `RocmDevice` reference (a static method on `RocmStorage` would create a
//! circular module import — see module-layout note in lib.rs).
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 3 — `BackendStorage` integration
//! - `rust-gpu-discipline` §3 — explicit `Result` on every GPU touch, no
//!   silent CPU fallback

use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::backend::BackendStorage;
use grim_tensor::error::{Error, Result};

// Re-exports used by the type's field types. The actual type declarations
// live in lib.rs and are reachable via the crate-root imports below.
use crate::{
    hipMemcpy, DType, HipMemcpyKind, QuantProvenance, RocmCachingAllocator, Shape, hipSuccess,
};
pub(crate) use crate::dtype_byte_size;

/// ROCm-side tensor storage. Holds a hipDeviceptr_t (as u64) plus shape/dtype/provenance metadata.
#[derive(Debug)]
pub struct RocmStorage {
    /// Opaque device pointer, stored as u64
    pub(crate) device_ptr: Option<u64>,
    pub(crate) bytes: usize,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) provenance: QuantProvenance,
    pub(crate) ordinal: usize,
    /// Back-reference to the owning device allocator; used by `Drop` to return the
    /// buffer to the free-list instead of calling `hipFree`.
    pub(crate) allocator: Arc<RocmCachingAllocator>,
}

impl RocmStorage {
    pub fn shape_metadata(&self) -> &Shape {
        &self.shape
    }

    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn device_ptr_is_valid(&self) -> bool {
        self.device_ptr.is_some()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for RocmStorage {
    fn drop(&mut self) {
        if let Some(ptr_val) = self.device_ptr {
            self.allocator.free(ptr_val as *mut c_void, self.bytes);
        }
    }
}

impl BackendStorage for RocmStorage {
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        if !self.device_ptr_is_valid() {
            return Err(Error::Backend(
                "RocmStorage has no valid device pointer".into(),
            ));
        }

        let mut host_data = vec![0.0f32; self.shape.elem_count()];
        let dev_ptr_void = self.device_ptr.unwrap() as *mut c_void;

        // Copy from device to host
        let res = unsafe {
            hipMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                dev_ptr_void,
                self.bytes,
                HipMemcpyKind::DeviceToHost,
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "hipMemcpyDtoH failed with error code {}",
                res
            )));
        }

        Ok(host_data)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
