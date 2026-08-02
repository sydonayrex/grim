//! `RocmStorage`: ROCm-side device buffer + metadata, plus its [see: `BackendStorage`, `alloc_gpu`]

use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::backend::BackendStorage;
use grim_tensor::dtype::KQuantScheme;
use grim_tensor::error::{Error, Result};

// Re-exports used by the type's field types. The actual type declarations
use crate::{
    DType, DTypeStorage, HipMemcpyKind, QuantProvenance, RocmCachingAllocator, Shape, check_hip,
    hipMallocManaged, hipMemPrefetchAsync, hipMemcpy, hipSuccess,
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
    /// Back-reference to the owning device allocator; used by `Drop` to return the [see: `hipFree`]
    pub(crate) allocator: Arc<RocmCachingAllocator>,
    /// Managed allocations may migrate between VRAM and host RAM under HIP.
    pub(crate) managed: bool,
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

    /// Raw HIP device pointer for integrations that invoke an external
    /// collective (for example RCCL) directly on this allocation.
    /// Callers must keep this storage alive and use the owning device ordinal.
    pub fn device_ptr_u64(&self) -> Option<u64> {
        self.device_ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether this storage uses HIP managed memory and may reside in host RAM.
    pub fn is_managed(&self) -> bool {
        self.managed
    }

    /// Allocates GPU memory via a caching allocator. Returns the storage on success. [see: `Arc<RocmCachingAllocator>`, `&RocmDevice`]
    pub fn alloc_gpu(
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        let bytes = shape.elem_count() * crate::dtype_byte_size(&dtype);
        if crate::memory::budget::use_managed_allocation(ordinal, bytes) {
            let mut ptr = std::ptr::null_mut();
            check_hip("hipMallocManaged", unsafe {
                hipMallocManaged(&mut ptr, bytes, 1)
            })?;
            return Ok(RocmStorage {
                device_ptr: Some(ptr as u64),
                bytes,
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
                ordinal,
                allocator: Arc::clone(allocator),
                managed: true,
            });
        }
        let dev_ptr_void = match allocator.alloc(bytes) {
            Ok(ptr) => ptr,
            Err(vram_error) => {
                // The free-memory probe is inherently racy: another stream or
                // rank may consume VRAM after the policy check. Preserve the
                // allocation request by falling back to managed memory before
                // surfacing the original device-allocation failure.
                let mut ptr = std::ptr::null_mut();
                if unsafe { hipMallocManaged(&mut ptr, bytes, 1) } == hipSuccess {
                    return Ok(RocmStorage {
                        device_ptr: Some(ptr as u64),
                        bytes,
                        shape: shape.clone(),
                        dtype,
                        provenance: QuantProvenance::GrimNative,
                        ordinal,
                        allocator: Arc::clone(allocator),
                        managed: true,
                    });
                }
                return Err(vram_error);
            }
        };

        Ok(RocmStorage {
            device_ptr: Some(dev_ptr_void as u64),
            bytes,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal,
            allocator: Arc::clone(allocator),
            managed: false,
        })
    }

    /// Copies data from host to GPU using the caching allocator + `hipMemcpy`. [see: `alloc_gpu`, `&[f32]`]
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

    /// Allocate HIP managed memory and initialize it from host data. The
    /// returned pointer is valid to ROCm kernels exactly like ordinary device
    /// storage; HIP may page it between VRAM and system RAM. This is the
    /// transparent overflow tier used for opt-in large model weights.
    pub fn copy_from_host_managed(
        host_data: &[f32],
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        if dtype.arith != grim_tensor::ArithType::F32 {
            return Err(Error::Unimplemented(
                "managed host-backed upload currently supports F32 weights only".into(),
            ));
        }
        let bytes = shape.elem_count() * crate::dtype_byte_size(&dtype);
        let mut ptr = std::ptr::null_mut();
        check_hip("hipMallocManaged", unsafe {
            hipMallocManaged(&mut ptr, bytes, 1)
        })?;
        let storage = Self {
            device_ptr: Some(ptr as u64),
            bytes,
            shape: shape.clone(),
            dtype: dtype.clone(),
            provenance: QuantProvenance::GrimNative,
            ordinal,
            allocator: Arc::clone(allocator),
            managed: true,
        };
        let result = unsafe {
            hipMemcpy(
                ptr,
                host_data.as_ptr() as *const c_void,
                bytes,
                HipMemcpyKind::HostToDevice,
            )
        };
        if result != hipSuccess {
            unsafe { let _ = crate::hipFree(ptr); }
            return Err(Error::Backend(format!(
                "hipMemcpyHostToDevice for managed storage failed with error code {result}"
            )));
        }
        Ok(storage)
    }

    /// Copies raw packed bytes (e.g. Q4_K, Q8_0, GPTQ) from host memory to GPU.
    pub fn copy_from_host_raw_bytes(
        host_bytes: &[u8],
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        let bytes = host_bytes.len();
        if crate::memory::budget::use_managed_allocation(ordinal, bytes) {
            let mut ptr = std::ptr::null_mut();
            check_hip("hipMallocManaged", unsafe {
                hipMallocManaged(&mut ptr, bytes, 1)
            })?;
            let result = unsafe {
                hipMemcpy(
                    ptr,
                    host_bytes.as_ptr() as *const c_void,
                    bytes,
                    HipMemcpyKind::HostToDevice,
                )
            };
            if result != hipSuccess {
                unsafe { let _ = crate::hipFree(ptr); }
                return Err(Error::Backend(format!(
                    "hipMemcpyHostToDevice for managed raw storage failed with error code {result}"
                )));
            }
            return Ok(RocmStorage {
                device_ptr: Some(ptr as u64),
                bytes,
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
                ordinal,
                allocator: Arc::clone(allocator),
                managed: true,
            });
        }
        let dev_ptr_void = match allocator.alloc(bytes) {
            Ok(ptr) => ptr,
            Err(vram_error) => {
                let mut ptr = std::ptr::null_mut();
                if unsafe { hipMallocManaged(&mut ptr, bytes, 1) } == hipSuccess {
                    let result = unsafe {
                        hipMemcpy(
                            ptr,
                            host_bytes.as_ptr() as *const c_void,
                            bytes,
                            HipMemcpyKind::HostToDevice,
                        )
                    };
                    if result == hipSuccess {
                        return Ok(RocmStorage {
                            device_ptr: Some(ptr as u64),
                            bytes,
                            shape: shape.clone(),
                            dtype,
                            provenance: QuantProvenance::GrimNative,
                            ordinal,
                            allocator: Arc::clone(allocator),
                            managed: true,
                        });
                    }
                    unsafe { let _ = crate::hipFree(ptr); }
                }
                return Err(vram_error);
            }
        };

        let upload_result = unsafe {
            hipMemcpy(
                dev_ptr_void,
                host_bytes.as_ptr() as *const c_void,
                bytes,
                HipMemcpyKind::HostToDevice,
            )
        };

        if upload_result != hipSuccess {
            allocator.free(dev_ptr_void, bytes);
            return Err(Error::Backend(format!(
                "hipMemcpyHostToDevice raw bytes failed with error code {}",
                upload_result
            )));
        }

        Ok(RocmStorage {
            device_ptr: Some(dev_ptr_void as u64),
            bytes,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal,
            allocator: Arc::clone(allocator),
            managed: false,
        })
    }
}

impl Drop for RocmStorage {
    fn drop(&mut self) {
        if let Some(ptr_val) = self.device_ptr {
            if self.managed {
                unsafe { let _ = crate::hipFree(ptr_val as *mut c_void); }
            } else {
                self.allocator.free(ptr_val as *mut c_void, self.bytes);
            }
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

    fn set_provenance(&mut self, provenance: QuantProvenance) {
        self.provenance = provenance;
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

        // Quantized storage (Q8_0, Q4K, …): the device buffer holds packed [see: `_`, `self.bytes`, `Vec<f32>`, `elem_count * 4`]
        if let DTypeStorage::KQuant(_scheme) = &self.dtype.storage {
            let mut raw = vec![0u8; self.bytes];
            check_hip("hipMemcpyDtoH (quantized)", unsafe {
                hipMemcpy(
                    raw.as_mut_ptr() as *mut c_void,
                    dev_ptr_void,
                    self.bytes,
                    HipMemcpyKind::DeviceToHost,
                )
            })?;
            return dequant_cpu(&raw, elem_count, &self.dtype);
        }

        // F16/BF16 storage: the device buffer holds 2-byte elements, but the
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

    fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }

    fn device_ordinal(&self) -> u32 {
        self.ordinal as u32
    }

    fn prefetch_to_device(&self) -> Result<()> {
        if !self.managed {
            return Ok(());
        }
        let ptr = self
            .device_ptr
            .ok_or_else(|| Error::Backend("managed storage has no device pointer".into()))?;
        check_hip("hipMemPrefetchAsync", unsafe {
            hipMemPrefetchAsync(
                ptr as *const c_void,
                self.bytes,
                self.ordinal as i32,
                std::ptr::null_mut(),
            )
        })
    }
}

/// Minimal host-side dequantizer for when a quantized tensor needs to be [see: `Vec<f32>`]
fn dequant_cpu(raw: &[u8], elem_count: usize, dtype: &DType) -> Result<Vec<f32>> {
    match &dtype.storage {
        DTypeStorage::KQuant(KQuantScheme::Q80) => {
            const QK8_0: usize = 32;
            let expected = (elem_count.div_ceil(QK8_0)) * (QK8_0 + 2);
            if raw.len() < expected {
                return Err(Error::Backend(format!(
                    "Q8_0 dequant: raw buffer {} bytes too small for {} elements (need {})",
                    raw.len(),
                    elem_count,
                    expected
                )));
            }
            let mut out = Vec::with_capacity(elem_count);
            for blk in raw.chunks_exact(QK8_0 + 2).take(elem_count.div_ceil(QK8_0)) {
                let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                let qs = &blk[2..2 + QK8_0];
                for &q in qs {
                    if out.len() < elem_count {
                        out.push(d * (q as f32));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::Backend(format!(
            "to_cpu_vec_f32: host dequant not yet implemented for {:?}",
            dtype.storage
        ))),
    }
}
