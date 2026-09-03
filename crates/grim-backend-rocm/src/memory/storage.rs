//! `RocmStorage`: ROCm-side device buffer + metadata, plus its [see: `BackendStorage`, `alloc_gpu`]

use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::backend::BackendStorage;
use grim_tensor::dtype::KQuantScheme;
use grim_tensor::error::{Error, Result};

// Re-exports used by the type's field types. The actual type declarations
use crate::{
    DType, DTypeStorage, HipMemcpyKind, QuantProvenance, RocmCachingAllocator, RocmDevice, Shape,
    check_hip, hipMallocManaged, hipMemPrefetchAsync, hipMemcpy, hipSuccess,
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

    /// Device pointer as a structured [`Result`], for the decode/prefill hot
    /// path — an un-uploaded (CPU-resident, `device_ptr == None`) tensor
    /// reaching a device op surfaces an error instead of panicking on
    /// `.device_ptr.unwrap()`.
    pub fn device_ptr_checked(&self) -> Result<u64> {
        self.device_ptr.ok_or_else(|| {
            Error::Backend(
                "expected device-resident ROCm storage (device_ptr is None) \
                 — a CPU tensor reached a device op"
                    .to_string(),
            )
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether this storage uses HIP managed memory and may reside in host RAM.
    pub fn is_managed(&self) -> bool {
        self.managed
    }

    /// Allocates GPU memory via a caching allocator with explicit byte count.
    pub fn alloc_gpu_with_bytes(
        shape: &Shape,
        dtype: DType,
        bytes: usize,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        if crate::memory::budget::use_managed_allocation(ordinal, bytes) {
            crate::memory::budget::note_managed_fallback(ordinal, bytes);
            // WI-M1 context discipline: `hipMallocManaged` binds the new
            // allocation to the CALLING THREAD's current device. Park on the
            // owning ordinal or a drifted thread materialises the buffer on
            // another device while `ordinal` still claims ownership.
            let _ctx = crate::device::util::DeviceGuard::set(ordinal as i32);
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
                let _ctx = crate::device::util::DeviceGuard::set(ordinal as i32);
                let mut ptr = std::ptr::null_mut();
                if unsafe { hipMallocManaged(&mut ptr, bytes, 1) } == hipSuccess {
                    crate::memory::budget::note_managed_fallback(ordinal, bytes);
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

    /// Allocates GPU memory via a caching allocator. Returns the storage on success. [see: `Arc<RocmCachingAllocator>`, `&RocmDevice`]
    pub fn alloc_gpu(
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        let bytes = shape.elem_count() * crate::dtype_byte_size(&dtype);
        Self::alloc_gpu_with_bytes(shape, dtype, bytes, allocator, ordinal)
    }

    /// Copies data from host to GPU using the caching allocator + `hipMemcpy`. [see: `alloc_gpu`, `&[f32]`]
    pub fn copy_from_host(
        host_data: &[f32],
        shape: &Shape,
        dtype: DType,
        allocator: &Arc<RocmCachingAllocator>,
        ordinal: usize,
    ) -> Result<Self> {
        if std::env::var("GRIM_ALLOC_TRACE").is_ok() {
            eprintln!(
                "[alloc-trace] copy_from_host bytes={} shape={:?}",
                host_data.len() * 4,
                shape.dims()
            );
        }
        eprintln!(
            "[grim-backend-rocm] copy_from_host: ENTER ordinal={} bytes={} shape={:?}",
            ordinal, host_data.len() * 4, shape.dims()
        );
        use std::io::Write;
        let _ = std::io::stderr().flush();
        let arith = dtype.arith;
        let mut storage = Self::alloc_gpu(shape, dtype, allocator, ordinal)?;
        eprintln!(
            "[grim-backend-rocm] copy_from_host: ALLOC OK storage.ordinal={} device_ptr={:?}",
            storage.ordinal, storage.device_ptr
        );
        let _ = std::io::stderr().flush();
        let dev_ptr_void = storage.device_ptr.unwrap() as *mut c_void;

        // WI-M1 context discipline: a synchronous `hipMemcpy` executes in the
        // calling thread's current device context. Pin the owning ordinal or a
        // drifted thread writes the tensor onto another device's memory.
        let _ctx = crate::device::util::DeviceGuard::set(ordinal as i32);
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

        eprintln!(
            "[grim-backend-rocm] copy_from_host: MEMCPY DONE ordinal={} result={}",
            ordinal, upload_result
        );
        let _ = std::io::stderr().flush();
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
        // WI-M1 context discipline: managed malloc + H2D fill both bind to the
        // calling thread's current device; pin the owning ordinal.
        let _ctx = crate::device::util::DeviceGuard::set(ordinal as i32);
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
            unsafe {
                let _ = crate::hipFree(ptr);
            }
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
        if std::env::var("GRIM_ALLOC_TRACE").is_ok() {
            eprintln!(
                "[alloc-trace] raw_bytes len={} shape={:?}",
                host_bytes.len(),
                shape.dims()
            );
        }
        let bytes = host_bytes.len();
        // WI-M1 context discipline: every branch below mixes a
        // thread-context-bound allocation (`hipMallocManaged` / allocator
        // miss) with a synchronous H2D fill. Pin the owning ordinal for the
        // whole seam so a drifted thread cannot land the quantized weights on
        // the wrong device — the exact producer of the ctx_dev=2 fault split.
        let _ctx = crate::device::util::DeviceGuard::set(ordinal as i32);
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
                unsafe {
                    let _ = crate::hipFree(ptr);
                }
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
                let _ctx = crate::device::util::DeviceGuard::set(ordinal as i32);
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
                    unsafe {
                        let _ = crate::hipFree(ptr);
                    }
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

    /// Read the raw device buffer bytes back to host.
    pub fn copy_to_host(&self) -> Result<Vec<u8>> {
        if !self.device_ptr_is_valid() {
            return Err(Error::Backend(
                "RocmStorage has no valid device pointer".into(),
            ));
        }
        let dev_ptr_void = self.device_ptr.unwrap() as *mut c_void;
        let _ctx = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut raw = vec![0u8; self.bytes];
        check_hip("hipMemcpyDtoH raw bytes", unsafe {
            hipMemcpy(
                raw.as_mut_ptr() as *mut c_void,
                dev_ptr_void,
                self.bytes,
                HipMemcpyKind::DeviceToHost,
            )
        })?;
        Ok(raw)
    }
}

impl Drop for RocmStorage {
    fn drop(&mut self) {
        if let Some(ptr_val) = self.device_ptr {
            // Pool returns (managed == false) do no HIP work, and real driver
            // releases are pinned inside `RocmCachingAllocator::free` /
            // `empty_cache` — so that branch stays guard-free to avoid an
            // extra hipGetDevice+hipSetDevice pair per drop (which once
            // widened a host-timing window in fused stream pipelines: mxfp4
            // rmsnorm→gemm→rope_kv parity flaked to all-zero outputs).
            // The managed branch issues a real `hipFree` and MUST pin the
            // owning ordinal — a drifted thread would otherwise free into
            // the wrong device's context.
            if self.managed {
                let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
                unsafe {
                    let _ = crate::hipFree(ptr_val as *mut c_void);
                }
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
        eprintln!(
            "[grim-backend-rocm] RocmStorage::to_cpu_vec_f32: ENTER self.ordinal={} bytes={} dtype={:?} ptr={:?}",
            self.ordinal, self.bytes, self.dtype, self.device_ptr
        );
        use std::io::Write;
        let _ = std::io::stderr().flush();
        let dev_ptr_void = self.device_ptr.unwrap() as *mut c_void;
        let elem_count = self.shape.elem_count();

        // WI-M1 context discipline: every DtoH branch below issues a
        // synchronous `hipMemcpy` against the calling thread's current device
        // context. Pin the owning ordinal so a drifted thread reads the right
        // allocation (and so the dequant launches below start from a sane
        // context).
        let _ctx = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        // Quantized storage (Q8_0, Q4K, …) or FloatPack (FP8): the device buffer holds packed bytes.
        // We copy the packed bytes DToH and dequantize them on GPU using the RocmDevice launchers
        // (or fallback to CPU dequant if GPU launch fails/unsupported).
        if matches!(
            &self.dtype.storage,
            DTypeStorage::KQuant(_) | DTypeStorage::FloatPack(_)
        ) {
            let mut raw = vec![0u8; self.bytes];
            check_hip("hipMemcpyDtoH (quantized/packed)", unsafe {
                hipMemcpy(
                    raw.as_mut_ptr() as *mut c_void,
                    dev_ptr_void,
                    self.bytes,
                    HipMemcpyKind::DeviceToHost,
                )
            })?;

            let dev = RocmDevice::shared(self.ordinal);
            let gpu_deq = match self.dtype.storage {
                DTypeStorage::KQuant(KQuantScheme::Q80) => {
                    dev.dequantize_q8_0_host(&raw, elem_count)
                }
                DTypeStorage::FloatPack(grim_tensor::FloatPackScheme::Fp8) => {
                    dev.dequantize_fp8_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                    dev.dequantize_q4k_host(&raw, elem_count)
                }
                DTypeStorage::FloatPack(grim_tensor::FloatPackScheme::MxFp4) => {
                    dev.dequantize_mxfp4_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::IQ4NL) => {
                    dev.dequantize_iq4nl_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::IQ4XS) => {
                    dev.dequantize_iq4xs_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => {
                    dev.dequantize_iq3xxs_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::IQ3S) => {
                    dev.dequantize_iq3s_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => {
                    dev.dequantize_iq2xxs_host(&raw, elem_count)
                }
                DTypeStorage::KQuant(KQuantScheme::IQ2XS) => {
                    dev.dequantize_iq2xs_host(&raw, elem_count)
                }
                _ => dequant_cpu(&raw, elem_count, &self.dtype),
            };

            let mut values = gpu_deq.or_else(|_| dequant_cpu(&raw, elem_count, &self.dtype))?;
            values.truncate(elem_count);
            return Ok(values);
        }

        // F16/BF16 storage: the device buffer holds 2-byte elements, but the
        let result_f32 = match self.dtype.arith {
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
            grim_tensor::ArithType::U8 => {
                let mut raw = vec![0u8; elem_count];
                check_hip("hipMemcpyDtoH (u8)", unsafe {
                    hipMemcpy(
                        raw.as_mut_ptr() as *mut c_void,
                        dev_ptr_void,
                        self.bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                Ok(raw.iter().map(|&b| b as f32).collect())
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
        };
        eprintln!(
            "[grim-backend-rocm] RocmStorage::to_cpu_vec_f32: EXIT ok self.ordinal={} out_len={}",
            self.ordinal,
            result_f32.as_ref().map(|v| v.len()).unwrap_or(0)
        );
        let _ = std::io::stderr().flush();
        result_f32
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
        let _ctx = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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
    let start = std::time::Instant::now();
    let result = match &dtype.storage {
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
                        out.push(d * (q as i8 as f32));
                    }
                }
            }
            Ok(out)
        }
        DTypeStorage::KQuant(KQuantScheme::Q2K) => grim_quant::dequant_q2k(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::Q3K) => grim_quant::dequant_q3k(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::Q4K) => grim_quant::dequant_q4k(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::Q5K) => grim_quant::dequant_q5k(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::Q6K) => grim_quant::dequant_q6k(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ4NL) => grim_quant::dequant_iq4nl(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ4XS) => grim_quant::dequant_iq4xs(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => grim_quant::dequant_iq3xxs(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ3S) => grim_quant::dequant_iq3s(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => grim_quant::dequant_iq2xxs(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ2XS) => grim_quant::dequant_iq2xs(raw, elem_count),
        DTypeStorage::KQuant(KQuantScheme::IQ2S) => grim_quant::dequant_iq2s(raw, elem_count),
        DTypeStorage::FloatPack(grim_tensor::FloatPackScheme::Fp8) => {
            if raw.len() < 4 + elem_count {
                return Err(Error::Backend(format!(
                    "FP8 dequant: raw buffer {} bytes too small for {} elements",
                    raw.len(),
                    elem_count
                )));
            }
            let scale_bits = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            let scale = f32::from_bits(scale_bits);
            let mut out = Vec::with_capacity(elem_count);
            for &byte in &raw[4..4 + elem_count] {
                let f_val = fp8_e4m3_to_f32(byte);
                out.push(scale * f_val);
            }
            Ok(out)
        }
        DTypeStorage::FloatPack(grim_tensor::FloatPackScheme::MxFp4) => {
            grim_quant::dequant_mxfp4(raw, elem_count)
        }
        _ => Err(Error::Backend(format!(
            "to_cpu_vec_f32: host dequant not yet implemented for {:?}",
            dtype.storage
        ))),
    };
    // Priority 3: surface progress for the host-side dequant fallback. Without
    // this, a legitimate ~10-minute load is silent between `[alias]` log lines
    // and is indistinguishable from a true hang. Borrowed from the `[grim]` log
    // convention used elsewhere in the load path.
    if let Ok(out) = &result {
        eprintln!(
            "[grim] Host dequantized {:?} ({} elements) in {:.2}s",
            dtype.storage,
            out.len(),
            start.elapsed().as_secs_f64()
        );
    }
    result
}

fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    if byte == 0x7F || byte == 0xFF {
        return f32::NAN;
    }
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;
    if exp == 0 {
        sign * (mant as f32) * (1.0 / 512.0)
    } else {
        sign * ((1 << 3) | mant) as f32 * (2.0f32.powi(exp as i32 - 7 - 3))
    }
}
