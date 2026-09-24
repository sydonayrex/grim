//! core_tensor_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, Shape};

use grim_backend_cpu::{CpuDevice, CpuStorage};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl CoreTensorOps for MetalDevice {
    /// Audit B5: device-side 2-D transpose via `grim_transpose_2d` - keeps LoRA A/B transposes resident on GPU.
    /// Non-Apple builds (and anything without a Metal context) fall back to the trait's host default.
    fn transpose_2d(
        &self,
        x: &dyn BackendStorage,
        rows: usize,
        cols: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if x.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(x.dtype())));
                }
                let ctx = inner.pipelines.clone();
                if x.shape().elem_count() != rows * cols {
                    return Err(Error::Shape(format!(
                        "transpose_2d: storage holds {} elements, expected {rows}×{cols}",
                        x.shape().elem_count()
                    )));
                }
                let n = rows * cols;
                let out_buf = ctx
                    .device
                    .newBufferWithLength_options(
                        (n * 4) as u64,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| Error::Backend("Metal transpose_2d: alloc out failed".into()))?;
                // Source buffer built from the raw little-endian bytes.
                let src_bytes_buf = {
                    let v = x.to_cpu_vec_f32()?;
                    let mut bytes = Vec::with_capacity(v.len() * 4);
                    for f in &v {
                        bytes.extend_from_slice(&f.to_le_bytes());
                    }
                    self.new_buffer_with_bytes(&bytes, BufferUsage::Shared)?
                };
                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Backend("Metal transpose_2d: encoder failed".into()))?;
                encoder.setComputePipelineState(&ctx.pipelines.transpose_2d);
                encoder.setBuffer_offset_atIndex(Some(&src_bytes_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
                let r = rows as u32;
                let c = cols as u32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &r as *const u32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                    encoder.setBytes_length_atIndex(
                        &c as *const u32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                }
                let grid = objc2_metal::MTLSize::new(((n + 255) / 256) as u64, 1, 1);
                let threads = objc2_metal::MTLSize::new(256, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
                encoder.endEncoding();
                cmd_buffer.commit();
                cmd_buffer.waitUntilCompleted();

                let ptr = out_buf.contents() as *const f32;
                let mut values = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), n);
                }
                let storage = self.from_cpu(&values, out_shape, x.dtype())?;
                return Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)));
            }
        }
        // Fallback: trait default host path.
        let v = x.to_cpu_vec_f32()?;
        if v.len() != rows * cols {
            return Err(Error::Shape(format!(
                "transpose_2d: storage holds {} elements, expected {rows}×{cols}",
                v.len()
            )));
        }
        let mut out = vec![0.0f32; v.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = v[r * cols + c];
            }
        }
        let storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let elem_count = shape.elem_count();
        let bytes = elem_count * dtype_byte_size(&dtype)?;
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use objc2_metal::MTResourceOptions;

                let buffer = inner
                    .device
                    .newBufferWithLength_options(
                        bytes as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::from(MetalError::AllocationFailed(
                            "Failed to allocate Metal buffer".into(),
                        ))
                    })?;

                let contents = buffer.contents();
                if !contents.is_null() {
                    unsafe {
                        std::ptr::write_bytes(contents, 0, bytes);
                    }
                }

                Ok(Box::new(MetalStorage {
                    buffer: Some(buffer),
                    data: None,
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            } else {
                Ok(Box::new(MetalStorage {
                    buffer: None,
                    data: Some(std::sync::Mutex::new(vec![0u8; bytes])),
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(vec![0u8; bytes]),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_with_op_internal(a, b, out, None)
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if a.dtype().arith != ArithType::F32 || b.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(a.dtype())));
                }
                self.run_elementwise(inner, &inner.pipelines.add, a, b, out)
            } else {
                run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                    cpu_dev.add(a_cpu, b_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.add(a_cpu, b_cpu, out_shape)
            })
        }
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if a.dtype().arith != ArithType::F32 || b.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(a.dtype())));
                }
                self.run_elementwise(inner, &inner.pipelines.mul, a, b, out)
            } else {
                run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                    cpu_dev.mul(a_cpu, b_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, a, b, out, |cpu_dev, a_cpu, b_cpu, out_shape| {
                cpu_dev.mul(a_cpu, b_cpu, out_shape)
            })
        }
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if gate.dtype().arith != ArithType::F32 || up.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(gate.dtype())));
                }
                self.run_elementwise(inner, &inner.pipelines.silu_mul, gate, up, out)
            } else {
                run_fallback_binary(self, gate, up, out, |cpu_dev, g_cpu, u_cpu, out_shape| {
                    cpu_dev.silu_mul(g_cpu, u_cpu, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, gate, up, out, |cpu_dev, g_cpu, u_cpu, out_shape| {
                cpu_dev.silu_mul(g_cpu, u_cpu, out_shape)
            })
        }
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if x.dtype().arith != ArithType::F32 || w.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(x.dtype())));
                }

                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rms_norm: input x is not MetalStorage".into())
                })?;
                let w_s = w.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rms_norm: input w is not MetalStorage".into())
                })?;
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("w has no GPU buffer".into()))?;

                let out_storage = self.zeros(out, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let total = out.elem_count();
                let row_len = x.shape().dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.rms_norm);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                let row_len_val = row_len;
                let eps_val = eps;
                let total_val = total as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &row_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &eps_val as *const f32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                run_fallback_binary(self, x, w, out, |cpu_dev, x_cpu, w_cpu, out_shape| {
                    cpu_dev.rms_norm(x_cpu, w_cpu, eps, out_shape)
                })
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            run_fallback_binary(self, x, w, out, |cpu_dev, x_cpu, w_cpu, out_shape| {
                cpu_dev.rms_norm(x_cpu, w_cpu, eps, out_shape)
            })
        }
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if x.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(x.dtype())));
                }

                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal softmax: input x is not MetalStorage".into())
                })?;
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;

                let out_storage = self.zeros(out, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let total = out.elem_count();
                let last_dim = out.dims().last().copied().unwrap_or(1) as i32;

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.softmax);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);

                let last_dim_val = last_dim;
                let total_val = total as i32;

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &last_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        2,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                let x_vec = x.to_cpu_vec_f32()?;
                tracing::warn!(
                    "Metal softmax: GPU path unavailable, falling back to CPU execution"
                );
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
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let x_vec = x.to_cpu_vec_f32()?;
            tracing::warn!("Metal softmax: non-Apple target, falling back to CPU execution");
            let cpu_dev = CpuDevice::new();
            let x_cpu = cpu_dev.from_cpu(&x_vec, x.shape(), x.dtype())?;
            let x_storage = x_cpu
                .as_any()
                .downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::Backend("Failed to downcast input x to CpuStorage".into()))?;
            let (res_storage, handle) = cpu_dev.softmax(x_storage, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, x.dtype())?;
            Ok((out_metal, handle))
        }
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if weight.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(weight.dtype())));
                }

                let w_s = weight
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("Metal embedding: weight is not MetalStorage".into())
                    })?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("weight has no GPU buffer".into()))?;

                // Create a temporary buffer for indices.
                let indices_bytes = indices.len().checked_mul(4).ok_or_else(|| {
                    Error::from(MetalError::AllocationFailed("Indices size overflow".into()))
                })?;
                let indices_u8 = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices_bytes)
                };
                let indices_buffer = self.new_buffer_with_bytes(indices_u8, BufferUsage::Shared)?;

                let out_storage = self.zeros(out, weight.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let embedding_dim = out.dims().last().copied().unwrap_or(1) as i32;
                let num_indices = indices.len() as i32;
                let total = out.elem_count();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.embedding);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&indices_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &embedding_dim as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_indices as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                }

                let threads_per_group = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                let w_vec = weight.to_cpu_vec_f32()?;
                tracing::warn!(
                    "Metal embedding: GPU path unavailable, falling back to CPU execution"
                );
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
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let w_vec = weight.to_cpu_vec_f32()?;
            tracing::warn!("Metal embedding: non-Apple target, falling back to CPU execution");
            let cpu_dev = CpuDevice::new();
            let w_cpu = cpu_dev.from_cpu(&w_vec, weight.shape(), weight.dtype())?;
            let w_storage = w_cpu
                .as_any()
                .downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::Backend("Failed to downcast weight to CpuStorage".into()))?;
            let (res_storage, handle) = cpu_dev.embedding(w_storage, indices, out)?;
            let res_vec = res_storage.to_cpu_vec_f32()?;
            let out_metal = self.from_cpu(&res_vec, out, weight.dtype())?;
            Ok((out_metal, handle))
        }
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let bytes = shape
            .elem_count()
            .checked_mul(dtype_byte_size(&dtype)?)
            .ok_or_else(|| {
                Error::from(MetalError::AllocationFailed("Buffer size overflow".into()))
            })?;
        if data.len() * 4 < bytes {
            return Err(Error::from(MetalError::DataMismatch(format!(
                "from_cpu: source slice ({} bytes) too small for destination ({} bytes)",
                data.len() * 4,
                bytes
            ))));
        }

        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref _inner) = self.inner {
                let data_bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
                let buffer = self.new_buffer_with_bytes(data_bytes, BufferUsage::Shared)?;

                Ok(Box::new(MetalStorage {
                    buffer: Some(buffer),
                    data: None,
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            } else {
                let mut fallback_data = vec![0u8; bytes];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        fallback_data.as_mut_ptr(),
                        bytes,
                    );
                }
                Ok(Box::new(MetalStorage {
                    buffer: None,
                    data: Some(std::sync::Mutex::new(fallback_data)),
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let mut fallback_data = vec![0u8; bytes];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    fallback_data.as_mut_ptr(),
                    bytes,
                );
            }
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(fallback_data),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }

    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        Ok(())
    }
}
