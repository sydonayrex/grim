//! `RocmStorage`: ROCm-side device buffer + metadata, plus its
//! `BackendStorage` trait impl and the canonical allocation helpers
//! (`alloc_gpu`, `copy_from_host`). The helpers take
//! `Arc<RocmCachingAllocator> + ordinal` rather than `&RocmDevice`
//! so this module stays free of any circular import with whatever
//! module ends up housing `RocmDevice`.
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

    /// Allocates GPU memory via a caching allocator. Returns the storage on success.
    ///
    /// Takes the components needed for allocation (`Arc<RocmCachingAllocator>` +
    /// ordinal) rather than a `&RocmDevice` reference. This breaks the circular
    /// module import between `memory::storage` and wherever `RocmDevice` lives,
    /// so the helper stays next to the `RocmStorage` type even after the device
    /// is extracted.
    pub fn alloc_gpu(
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        let bytes = shape.elem_count() * crate::dtype_byte_size(&dtype);
        let dev_ptr_void = allocator.alloc(bytes)?;

        Ok(RocmStorage {
            device_ptr: Some(dev_ptr_void as u64),
            bytes,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal,
            allocator: Arc::clone(allocator),
        })
    }

    /// Copies data from host to GPU using the caching allocator + `hipMemcpy`.
    ///
    /// Same parameter-shape rationale as [`alloc_gpu`]: pulls the bytes from a
    /// `&[f32]` (the only dtype currently wired through), routing the
    /// allocation through the caching allocator passed in.
    pub fn copy_from_host(
        host_data: &[f32],
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        let mut storage = Self::alloc_gpu(shape, dtype, allocator, ordinal)?;

        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;
        let res = unsafe {
            hipMemcpy(
                dev_ptr_void,
                host_data.as_ptr() as *const c_void,
                storage.bytes,
                HipMemcpyKind::HostToDevice,
            )
        };
        if res != hipSuccess {
            // Return the buffer to the pool (Drop would also do this, but be explicit).
            unsafe {
                storage.allocator.free(dev_ptr_void, storage.bytes);
            }
            storage.device_ptr = None;
            return Err(Error::Backend(format!(
                "hipMemcpyHostToDevice failed with error code {}",
                res
            )));
        }

        Ok(storage)
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
