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
    check_hip, hipMemcpy, DType, HipMemcpyKind, QuantProvenance, RocmCachingAllocator, Shape,
    hipSuccess,
};

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
        let arith = dtype.arith;
        let mut storage = Self::alloc_gpu(shape, dtype, allocator, ordinal)?;
        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;

        // F16/BF16: the host provides f32 values but the device buffer holds
        // 2-byte elements. Convert before uploading so the kernel reads valid
        // quantized data, not raw f32 bit-pairs misinterpreted as f16.
        let upload_result = match arith {
            grim_tensor::ArithType::F16 => {
                let f16_vec: Vec<half::f16> =
                    host_data.iter().map(|&f| half::f16::from_f32(f)).collect();
                let bytes = f16_vec.len() * 2;
                unsafe {
                    hipMemcpy(
                        dev_ptr_void,
                        f16_vec.as_ptr() as *const c_void,
                        bytes,
                        HipMemcpyKind::HostToDevice,
                    )
                }
            }
            grim_tensor::ArithType::BF16 => {
                let bf16_vec: Vec<half::bf16> =
                    host_data.iter().map(|&f| half::bf16::from_f32(f)).collect();
                let bytes = bf16_vec.len() * 2;
                unsafe {
                    hipMemcpy(
                        dev_ptr_void,
                        bf16_vec.as_ptr() as *const c_void,
                        bytes,
                        HipMemcpyKind::HostToDevice,
                    )
                }
            }
            // F32 and integer types: direct memcpy (source is already f32 bytes).
            _ => unsafe {
                hipMemcpy(
                    dev_ptr_void,
                    host_data.as_ptr() as *const c_void,
                    storage.bytes,
                    HipMemcpyKind::HostToDevice,
                )
            },
        };

        if upload_result != hipSuccess {
            storage.allocator.free(dev_ptr_void, storage.bytes);
            storage.device_ptr = None;
            return Err(Error::Backend(format!(
                "hipMemcpyHostToDevice failed with error code {}",
                upload_result
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

        let dev_ptr_void = self.device_ptr.unwrap() as *mut c_void;
        let elem_count = self.shape.elem_count();

        // F16/BF16 storage: the device buffer holds 2-byte elements, but the
        // caller expects f32. We can't just memcpy into a Vec<f32> (that would
        // reinterpret pairs of f16 values as single f32 values — a silent
        // correctness bug). Instead, read the raw 2-byte elements into a byte
        // buffer, then convert each to f32.
        match self.dtype.arith {
            grim_tensor::ArithType::F16 => {
                let mut raw = vec![0u8; elem_count * 2];
                check_hip("hipMemcpyDtoH (f16)", unsafe {
                    hipMemcpy(
                        raw.as_mut_ptr() as *mut c_void,
                        dev_ptr_void,
                        self.bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                // Reinterpret the byte buffer as f16 (little-endian) and convert.
                let f16_slice: &[half::f16] = unsafe {
                    std::slice::from_raw_parts(raw.as_ptr() as *const half::f16, elem_count)
                };
                Ok(f16_slice.iter().map(|&h| h.to_f32()).collect())
            }
            grim_tensor::ArithType::BF16 => {
                let mut raw = vec![0u8; elem_count * 2];
                check_hip("hipMemcpyDtoH (bf16)", unsafe {
                    hipMemcpy(
                        raw.as_mut_ptr() as *mut c_void,
                        dev_ptr_void,
                        self.bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                let bf16_slice: &[half::bf16] = unsafe {
                    std::slice::from_raw_parts(raw.as_ptr() as *const half::bf16, elem_count)
                };
                Ok(bf16_slice.iter().map(|&b| b.to_f32()).collect())
            }
            // F32 and integer types: direct memcpy into f32 buffer.
            _ => {
                let mut host_data = vec![0.0f32; elem_count];
                check_hip("hipMemcpyDtoH", unsafe {
                    hipMemcpy(
                        host_data.as_mut_ptr() as *mut c_void,
                        dev_ptr_void,
                        self.bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                Ok(host_data)
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
