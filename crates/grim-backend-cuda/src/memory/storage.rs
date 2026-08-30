//! Device storage abstraction for GPU VRAM buffers on CUDA devices.

use std::ffi::c_void;
use grim_tensor::dtype::{ArithType, BlockDtype, DType, FloatPackScheme, KQuantScheme, QuantProvenance, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::cuda_device::CudaDevice;
use crate::device::handles::{
    cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice,
    cudaMemset, cudaSetDevice, cudaSuccess,
};

/// Returns the byte size of a DType.
pub(crate) fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 | ArithType::BF16 => 2,
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}

/// Stages a CUDA-resident packed quantized buffer to a host `Vec<u8>` via `cudaMemcpy` (D→H).
pub(crate) fn stage_packed_bytes(packed: &CudaStorage) -> Result<Vec<u8>> {
    let dev_ptr = CudaDevice::dev_ptr_or_err("stage_packed_bytes", packed)? as *const c_void;
    let mut raw = vec![0u8; packed.bytes];
    let res = unsafe {
        cudaMemcpy(
            raw.as_mut_ptr() as *mut c_void,
            dev_ptr as *mut c_void,
            packed.bytes,
            cudaMemcpyDeviceToHost,
        )
    };
    if res != cudaSuccess {
        return Err(Error::Backend(format!(
            "stage_packed_bytes: cudaMemcpy D→H failed with error code {}",
            res
        )));
    }
    Ok(raw)
}

/// Reads a length-prefixed segment from `bytes` starting at `*cursor`.
pub(crate) fn read_length_prefixed(bytes: &[u8], cursor: &mut usize) -> Result<Vec<u8>> {
    if bytes.len() < *cursor + 8 {
        return Err(Error::Backend(
            "read_length_prefixed: truncated segment length prefix".into(),
        ));
    }
    let len = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap()) as usize;
    *cursor += 8;
    if bytes.len() < *cursor + len {
        return Err(Error::Backend(format!(
            "read_length_prefixed: truncated segment (expected {len} bytes)"
        )));
    }
    let segment = bytes[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Ok(segment)
}

/// Host-side dequantization of CUDA-resident packed storage.
pub(crate) fn cuda_dequant_quantized_storage(
    b_bytes: &[u8],
    _b_scales: Option<&[f32]>,
    elem_count: usize,
    dtype: &DType,
) -> Result<Vec<f32>> {
    match &dtype.storage {
        DTypeStorage::KQuant(scheme) => match scheme {
            KQuantScheme::Q2K => grim_quant::dequant_q2k(b_bytes, elem_count),
            KQuantScheme::Q3K => grim_quant::dequant_q3k(b_bytes, elem_count),
            KQuantScheme::Q4K => grim_quant::dequant_q4k(b_bytes, elem_count),
            KQuantScheme::Q5K => grim_quant::dequant_q5k(b_bytes, elem_count),
            KQuantScheme::Q6K => grim_quant::dequant_q6k(b_bytes, elem_count),
            KQuantScheme::Q80 => grim_quant::dequant_q80(b_bytes, elem_count),
            KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(b_bytes, elem_count),
            KQuantScheme::IQ4XS => grim_quant::dequant_iq4xs(b_bytes, elem_count),
            KQuantScheme::IQ3XXS => grim_quant::dequant_iq3xxs(b_bytes, elem_count),
            KQuantScheme::IQ3S => grim_quant::dequant_iq3s(b_bytes, elem_count),
            KQuantScheme::IQ2XXS => grim_quant::dequant_iq2xxs(b_bytes, elem_count),
            KQuantScheme::IQ2XS => grim_quant::dequant_iq2xs(b_bytes, elem_count),
            KQuantScheme::IQ2S => grim_quant::dequant_iq2s(b_bytes, elem_count),
        },
        DTypeStorage::FloatPack(scheme) => match scheme {
            FloatPackScheme::Fp4 => grim_quant::dequant_fp4(b_bytes, elem_count),
            FloatPackScheme::Nf4 => grim_quant::dequant_nf4(b_bytes, elem_count),
            FloatPackScheme::Fp8 => grim_quant::dequant_fp8(b_bytes, elem_count),
            FloatPackScheme::MxFp4 => grim_quant::dequant_mxfp4(b_bytes, elem_count),
            FloatPackScheme::MxFp8 => grim_quant::dequant_mxfp8(b_bytes, elem_count),
        },
        DTypeStorage::Block(bd) => match bd {
            BlockDtype::Fp4 | BlockDtype::Fp4Block16 => {
                grim_quant::dequant_fp4_block16(b_bytes, elem_count)
            }
            BlockDtype::Nf4 => grim_quant::dequant_nf4(b_bytes, elem_count),
            BlockDtype::Fp8 | BlockDtype::Fp8Block16 => {
                grim_quant::dequant_fp8_block16(b_bytes, elem_count)
            }
        },
        DTypeStorage::ResidualPacked(cfg) => Err(Error::Unimplemented(format!(
            "quantized_matmul_backward_dx: ResidualPacked (bpw {}) host dequant not implemented; \
             this layout requires a fused device kernel",
            cfg.bpw
        ))),
        DTypeStorage::GroupInt(_) => Err(Error::Unimplemented(
            "quantized_matmul_backward_dx: GroupInt storage is dequantized to F32 at load time \
             on CUDA and does not reach the fused path"
                .into(),
        )),
        DTypeStorage::Native => Err(Error::Backend(format!(
            "quantized_matmul_backward_dx: expected quantized b, got Native ({:?})",
            dtype
        ))),
        _ => Err(Error::Unimplemented(format!(
            "quantized_matmul_backward_dx: host dequant for storage {:?} not implemented",
            dtype.storage
        ))),
    }
}

/// CUDA tensor storage.
#[derive(Debug)]
pub struct CudaStorage {
    pub(crate) device_ptr: Option<u64>,
    pub(crate) bytes: usize,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) provenance: QuantProvenance,
    pub(crate) ordinal: usize,
}

impl CudaStorage {
    /// Allocates GPU memory sized to exactly `byte_len` bytes.
    pub fn alloc_gpu_bytes(
        shape: &Shape,
        dtype: DType,
        byte_len: usize,
        device_ordinal: usize,
    ) -> Result<Self> {
        let select_res = unsafe { cudaSetDevice(device_ordinal as i32) };
        if select_res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                device_ordinal
            )));
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let mut res = unsafe { cudaMalloc(&mut dev_ptr, byte_len) };
        if res != cudaSuccess {
            unsafe {
                let _ = crate::device::handles::cudaDeviceSynchronize();
            }
            res = unsafe { cudaMalloc(&mut dev_ptr, byte_len) };
        }
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMalloc failed to allocate {} bytes with error {}",
                byte_len, res
            )));
        }

        Ok(Self {
            device_ptr: Some(dev_ptr as u64),
            bytes: byte_len,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal: device_ordinal,
        })
    }

    /// Allocates GPU memory on a CUDA device.
    pub fn alloc_gpu(shape: &Shape, dtype: DType, device_ordinal: usize) -> Result<Self> {
        let bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype))
            .ok_or_else(|| {
                Error::Backend(format!(
                    "alloc_gpu: byte count overflow for shape {:?} dtype {:?}",
                    shape, dtype
                ))
            })?;

        let select_res = unsafe { cudaSetDevice(device_ordinal as i32) };
        if select_res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                device_ordinal
            )));
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let mut res = unsafe { cudaMalloc(&mut dev_ptr, bytes) };
        if res != cudaSuccess {
            unsafe {
                let _ = crate::device::handles::cudaDeviceSynchronize();
            }
            res = unsafe { cudaMalloc(&mut dev_ptr, bytes) };
        }
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMalloc failed to allocate {} bytes with error {}",
                bytes, res
            )));
        }

        Ok(Self {
            device_ptr: Some(dev_ptr as u64),
            bytes,
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal: device_ordinal,
        })
    }

    /// Zeroes the backing GPU buffer.
    pub fn fill_zeroes(&self) -> Result<()> {
        let dev_ptr = match self.device_ptr {
            Some(p) => p as *mut c_void,
            None => return Ok(()),
        };
        if unsafe { cudaSetDevice(self.ordinal as i32) } != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaSetDevice failed for device {}",
                self.ordinal
            )));
        }
        let res = unsafe { cudaMemset(dev_ptr, 0, self.bytes) };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemset failed to zero {} bytes (err {})",
                self.bytes, res
            )));
        }
        Ok(())
    }

    /// Copies host data to GPU via cudaMemcpy.
    pub fn copy_from_host(
        host_data: &[f32],
        shape: &Shape,
        dtype: DType,
        device_ordinal: usize,
    ) -> Result<Self> {
        let storage = Self::alloc_gpu(shape, dtype, device_ordinal)?;
        let dev_ptr = storage.device_ptr.unwrap() as *mut c_void;

        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                host_data.as_ptr() as *const c_void,
                storage.bytes,
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            unsafe {
                let _ = cudaFree(dev_ptr);
            }
            return Err(Error::Backend(format!(
                "cudaMemcpyHostToDevice failed with error code {}",
                res
            )));
        }

        Ok(storage)
    }

    /// Copies raw packed bytes from host memory to GPU.
    pub fn copy_from_host_raw_bytes(
        host_bytes: &[u8],
        shape: &Shape,
        dtype: DType,
        device_ordinal: usize,
    ) -> Result<Self> {
        let storage = Self::alloc_gpu_bytes(shape, dtype, host_bytes.len(), device_ordinal)?;
        let dev_ptr = storage.device_ptr.ok_or_else(|| {
            Error::Backend("copy_from_host_raw_bytes: device_ptr is null after alloc".into())
        })? as *mut c_void;

        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                host_bytes.as_ptr() as *const c_void,
                host_bytes.len(),
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            unsafe {
                let _ = cudaFree(dev_ptr);
            }
            return Err(Error::Backend(format!(
                "cudaMemcpyHostToDevice (raw bytes) failed with error code {}",
                res
            )));
        }

        Ok(storage)
    }

    /// Returns the tensor shape.
    pub fn shape_metadata(&self) -> &Shape {
        &self.shape
    }

    /// Returns the device ordinal.
    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns the device pointer if allocated.
    pub fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }

    /// Returns the storage size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Download the raw packed bytes into a host `Vec<u8>`.
    pub fn copy_to_host_raw_bytes(&self) -> Result<Vec<u8>> {
        let dev_ptr = self
            .device_ptr
            .ok_or_else(|| Error::Backend("CudaStorage has no valid device pointer".into()))?
            as *const c_void;
        let mut host = vec![0u8; self.bytes];
        let res = unsafe {
            cudaMemcpy(
                host.as_mut_ptr() as *mut c_void,
                dev_ptr,
                self.bytes,
                cudaMemcpyDeviceToHost,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpyDeviceToHost failed with error code {}",
                res
            )));
        }
        Ok(host)
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        if let Some(ptr_val) = self.device_ptr {
            if ptr_val != 0 {
                unsafe {
                    let _ = cudaFree(ptr_val as *mut c_void);
                }
            }
        }
    }
}

impl BackendStorage for CudaStorage {
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copies GPU buffer to host as F32 vector.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        let dev_ptr = self
            .device_ptr
            .ok_or_else(|| Error::Backend("CudaStorage has no valid device pointer".into()))?
            as *mut c_void;

        let elem_count = self.shape.elem_count();

        if self.dtype.is_quantized() {
            if self.device_ptr.is_some() {
                if let Ok(dev) = CudaDevice::new(self.ordinal) {
                    if let Ok(f32_storage) = dev.dequantize_on_device(self) {
                        let mut host_data = vec![0.0f32; elem_count];
                        let res = unsafe {
                            cudaMemcpy(
                                host_data.as_mut_ptr() as *mut c_void,
                                CudaDevice::dev_ptr_or_err("to_cpu_vec_f32(gpu)", &f32_storage)?
                                    as *mut c_void,
                                f32_storage.bytes,
                                cudaMemcpyDeviceToHost,
                            )
                        };
                        if res == cudaSuccess {
                            drop(f32_storage);
                            return Ok(host_data);
                        }
                    }
                }
            }

            let mut raw = vec![0u8; self.bytes];
            let res = unsafe {
                cudaMemcpy(
                    raw.as_mut_ptr() as *mut c_void,
                    dev_ptr,
                    self.bytes,
                    cudaMemcpyDeviceToHost,
                )
            };
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMemcpyDeviceToHost (quantized) failed with error code {}",
                    res
                )));
            }
            let b_scales = <CudaStorage as BackendStorage>::quant_scales(self);
            return cuda_dequant_quantized_storage(&raw, b_scales, elem_count, &self.dtype);
        }

        let mut host_data = vec![0.0f32; elem_count];
        let res = unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                dev_ptr,
                self.bytes,
                cudaMemcpyDeviceToHost,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpyDeviceToHost failed with error code {}",
                res
            )));
        }

        Ok(host_data)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device_ordinal(&self) -> u32 {
        self.ordinal as u32
    }

    fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }
}
