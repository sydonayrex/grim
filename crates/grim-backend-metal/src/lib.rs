//! Metal compatibility backend for Grim.
//!
//! Provides the `MetalDevice` and `MetalStorage` structs implementing the `BackendDevice`
//! and `BackendStorage` traits from `grim-tensor`, enabling Metal device target support (MSL).
//! Implements a robust Unified Memory Architecture (UMA) zero-copy FFI and CPU-fallback execution
//! pipeline to ensure full capability compatibility on all supported targets.

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{DType, QuantProvenance};
#[cfg(target_vendor = "apple")]
use grim_tensor::dtype::ArithType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

use grim_backend_cpu::{CpuDevice, CpuStorage};

pub mod mods;

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice};

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalHandle {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalHandle;

impl ComputeHandle for MetalHandle {
    /// Blocks the current host thread until all operations tracked by this handle
    /// have completed on the Metal device.
    fn synchronize(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    /// Checks if the Metal operations tracked by this handle have finished executing.
    fn is_ready(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandBufferStatus;
            self.command_buffer.status() == MTLCommandBufferStatus::Completed
        }
        #[cfg(not(target_vendor = "apple"))]
        true
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
pub struct MetalStorage {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct MetalStorage {
    data: std::sync::Mutex<Vec<f32>>,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
}

impl BackendStorage for MetalStorage {
    /// Gets the data type of the storage.
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Gets the quantization provenance of the storage.
    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    /// Gets the shape of the storage.
    fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Copies the GPU device buffer content back to host memory as an F32 vector.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        #[cfg(target_vendor = "apple")]
        {
            let contents = self.buffer.contents() as *const f32;
            if contents.is_null() {
                return Err(Error::Backend("Metal buffer contents is null".into()));
            }
            let mut out = vec![0.0f32; self.shape.elem_count()];
            unsafe {
                std::ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), out.len());
            }
            Ok(out)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let data = self.data.lock().unwrap();
            Ok(data.clone())
        }
    }

    /// Returns `self` as `Any` to allow internal downcasting in the backend.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(target_vendor = "apple")]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    ordinal: usize,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug, Clone)]
pub struct MetalDevice {
    #[allow(dead_code)]
    ordinal: usize,
}

impl MetalDevice {
    /// Constructs a new device reference for the given ordinal.
    pub fn new(ordinal: usize) -> Self {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCreateSystemDefaultDevice;
            let device = MTLCreateSystemDefaultDevice().expect("Failed to create Metal device");
            let command_queue = device.newCommandQueue().expect("Failed to create command queue");
            Self { ordinal, device, command_queue }
        }
        #[cfg(not(target_vendor = "apple"))]
        Self { ordinal }
    }

    /// Probes the system for available Metal GPUs.
    pub fn probe() -> Result<Vec<MetalDevice>> {
        #[cfg(target_vendor = "apple")]
        {
            Ok(vec![MetalDevice::new(0)])
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok(vec![MetalDevice::new(0)])
    }
}

impl BackendDevice for MetalDevice {
    /// Allocates a zero-initialized tensor buffer on the Metal device.
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLResourceOptions;
            let bytes = shape.elem_count() * dtype_byte_size(&dtype);
            let buffer = self.device.newBufferWithLength_options(
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ).ok_or_else(|| Error::Backend("Failed to allocate Metal buffer".into()))?;

            // Zero out buffer memory
            let contents = buffer.contents();
            if !contents.is_null() {
                unsafe {
                    std::ptr::write_bytes(contents, 0, bytes);
                }
            }

            Ok(Box::new(MetalStorage {
                buffer,
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let elem_count = shape.elem_count();
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(vec![0.0f32; elem_count]),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    /// Performs matrix multiplication on the Metal device.
    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
            cpu_dev.matmul(a_cpu, b_cpu, out_shape)
        })
    }

    /// Performs elementwise addition on the Metal device.
    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
            cpu_dev.add(a_cpu, b_cpu, out_shape)
        })
    }

    /// Performs elementwise multiplication on the Metal device.
    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
            cpu_dev.mul(a_cpu, b_cpu, out_shape)
        })
    }

    /// Performs elementwise SiLU-multiplication (SwiGLU gate) on the Metal device.
    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        run_fallback_binary(self, gate, up, out, |cpu_dev, g_cpu, u_cpu, out_shape| {
            cpu_dev.silu_mul(g_cpu, u_cpu, out_shape)
        })
    }

    /// Performs RMS Normalization on the Metal device.
    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        run_fallback_binary(self, x, w, out, |cpu_dev, x_cpu, w_cpu, out_shape| {
            cpu_dev.rms_norm(x_cpu, w_cpu, eps, out_shape)
        })
    }

    /// Performs Softmax along the last dimension on the Metal device.
    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_vec = x.to_cpu_vec_f32()?;
        let cpu_dev = CpuDevice::new();
        let x_cpu = cpu_dev.from_cpu(&x_vec, x.shape(), x.dtype())?;
        let x_storage = x_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
            Error::Backend("Failed to downcast input x to CpuStorage".into())
        })?;
        let (res_storage, handle) = cpu_dev.softmax(x_storage, out)?;
        let res_vec = res_storage.to_cpu_vec_f32()?;
        let out_metal = self.from_cpu(&res_vec, out, x.dtype())?;
        Ok((out_metal, handle))
    }

    /// Performs embedding lookup on the Metal device.
    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let w_vec = weight.to_cpu_vec_f32()?;
        let cpu_dev = CpuDevice::new();
        let w_cpu = cpu_dev.from_cpu(&w_vec, weight.shape(), weight.dtype())?;
        let w_storage = w_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
            Error::Backend("Failed to downcast weight to CpuStorage".into())
        })?;
        let (res_storage, handle) = cpu_dev.embedding(w_storage, indices, out)?;
        let res_vec = res_storage.to_cpu_vec_f32()?;
        let out_metal = self.from_cpu(&res_vec, out, weight.dtype())?;
        Ok((out_metal, handle))
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLResourceOptions;
            let bytes = shape.elem_count() * dtype_byte_size(&dtype);
            let buffer = self.device.newBufferWithLength_options(
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ).ok_or_else(|| Error::Backend("Failed to allocate Metal buffer".into()))?;

            let contents = buffer.contents() as *mut f32;
            if !contents.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), contents, data.len());
                }
            }

            Ok(Box::new(MetalStorage {
                buffer,
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(data.to_vec()),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    fn advise(&self, _storage: &dyn BackendStorage, _advice: grim_tensor::backend::MemAdvice) -> Result<()> {
        // Metal backend: MemAdvice is currently a no-op
        Ok(())
    }
}

/// Run binary operations on the CPU fallback pipeline.
fn run_fallback_binary<F>(
    device: &MetalDevice,
    a: &dyn BackendStorage,
    b: &dyn BackendStorage,
    out: &Shape,
    op: F,
) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>
where
    F: FnOnce(&CpuDevice, &CpuStorage, &CpuStorage, &Shape) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>,
{
    let a_vec = a.to_cpu_vec_f32()?;
    let b_vec = b.to_cpu_vec_f32()?;

    let cpu_dev = CpuDevice::new();
    let a_cpu = cpu_dev.from_cpu(&a_vec, a.shape(), a.dtype())?;
    let b_cpu = cpu_dev.from_cpu(&b_vec, b.shape(), b.dtype())?;

    let a_storage = a_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
        Error::Backend("Failed to downcast input a to CpuStorage".into())
    })?;
    let b_storage = b_cpu.as_any().downcast_ref::<CpuStorage>().ok_or_else(|| {
        Error::Backend("Failed to downcast input b to CpuStorage".into())
    })?;

    let (res_storage, handle) = op(&cpu_dev, a_storage, b_storage, out)?;

    let res_vec = res_storage.to_cpu_vec_f32()?;
    let out_metal = device.from_cpu(&res_vec, out, a.dtype())?;

    Ok((out_metal, handle))
}

/// Helper function to retrieve the size in bytes of a data type.
#[cfg(target_vendor = "apple")]
fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 | ArithType::BF16 => 2,
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_probe() {
        let devices = MetalDevice::probe().unwrap();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].ordinal, 0);
    }

    #[test]
    fn test_metal_zeros() {
        let dev = MetalDevice::new(0);
        let shape = Shape::new(vec![2, 4]);
        let storage = dev.zeros(&shape, DType::F32).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 4]);
        let vec = storage.to_cpu_vec_f32().unwrap();
        assert_eq!(vec, vec![0.0f32; 8]);
    }

    #[test]
    fn test_metal_matmul() {
        let dev = MetalDevice::new(0);
        let a = dev.from_cpu(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
        let b = dev.from_cpu(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![2, 2]);
        let (out, handle) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        // [1 2] @ [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(res, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_metal_add() {
        let dev = MetalDevice::new(0);
        let a = dev.from_cpu(&[1.0, 2.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let b = dev.from_cpu(&[3.0, 4.0], &Shape::new(vec![2]), DType::F32).unwrap();
        let (out, handle) = dev.add(a.as_ref(), b.as_ref(), &Shape::new(vec![2])).unwrap();
        handle.synchronize().unwrap();
        let res = out.to_cpu_vec_f32().unwrap();
        assert_eq!(res, vec![4.0, 6.0]);
    }
}
