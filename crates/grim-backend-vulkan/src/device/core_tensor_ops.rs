//! `CoreTensorOps` implementation for VulkanDevice.
//! Extracted from lib.rs (modularization): trait impls live in `device/`, dispatch plumbing in `kernel.rs`, buffers in.

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendStorage, CoreTensorOps, Shape};

use crate::context::global_context;
use crate::ffi::*;
use crate::kernel::{VulkanKernel, push_params, run_compute_shader, spirv_for};
use crate::{VulkanDevice, VulkanStorage, f32_to_bf16_to_f32};

impl CoreTensorOps for VulkanDevice {
    /// Tier A: delegate to the existing rms_norm kernel.
    /// (No separate in-place shader: the trait contract only requires the HANDLE semantics; the allocation-free in-place.
    fn rms_norm_inplace(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<Box<dyn ComputeHandle>> {
        let (_storage, handle) = self.rms_norm(x, weight, eps, out)?;
        Ok(handle)
    }

    /// Tier A: solution_index has no Vulkan analogue (rocBLAS solver hint);
    /// fall through to the standard matmul.
    fn matmul_with_solution(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
        _solution_index: i32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul(a, b, out)
    }

    /// B5: device-side 2-D transpose - `[rows, cols] -> [cols, rows]` via the `grim_transpose_2d` compute shader.
    /// Eliminates the host round-trip the `lora_accumulate` default previously needed to transpose its A/B operands on.
    fn transpose_2d(
        &self,
        x: &dyn BackendStorage,
        rows: usize,
        cols: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan transpose_2d x is not VulkanStorage".into()))?;
        if x.shape().elem_count()
            != rows
                .checked_mul(cols)
                .ok_or_else(|| Error::Shape("Vulkan transpose_2d: rows*cols overflow".into()))?
        {
            return Err(Error::Shape(format!(
                "Vulkan transpose_2d: storage holds {} elements, expected {rows}×{cols}",
                x.shape().elem_count()
            )));
        }
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Transpose2d).to_vec();
        let buffers = [x_s.buffer, out_storage.buffer];
        let n = rows * cols;
        let grid_x = n.div_ceil(256) as u32;

        let push = push_params(rows as u32, cols as u32, 0, 0, 0, 0.0);
        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage = VulkanStorage::alloc_gpu(shape, dtype, ctx.device, ctx.physical_device)?;

        // Map and zero-fill
        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed with status {}",
                res
            )));
        }

        unsafe {
            std::ptr::write_bytes(mapped, 0, storage.bytes);
            vkUnmapMemory(ctx.device, storage.memory);
        }

        Ok(Box::new(storage))
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, None)
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan add: input a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan add: input b is not VulkanStorage".into()))?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let size = out.elem_count();
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Add).to_vec();

        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan add GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan mul: input a is not VulkanStorage".into()))?;
        let b_s = b
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan mul: input b is not VulkanStorage".into()))?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let size = out.elem_count();
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Mul).to_vec();

        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan mul GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_s = gate
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan silu_mul: input gate is not VulkanStorage".into())
            })?;
        let up_s = up.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan silu_mul: input up is not VulkanStorage".into())
        })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let size = out.elem_count();
        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::SiluMul).to_vec();

        let buffers = [gate_s.buffer, up_s.buffer, out_storage.buffer];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, 0, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan silu_mul GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan rms_norm: input x is not VulkanStorage".into())
        })?;
        let w_s = weight
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rms_norm: input weight is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let size = out.elem_count();
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::RmsNorm).to_vec();

        let buffers = [x_s.buffer, w_s.buffer, out_storage.buffer];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, eps);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan rms_norm GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn softmax(
        &self,

        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan softmax: input x is not VulkanStorage".into()))?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let size = out.elem_count();
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Softmax).to_vec();

        let buffers = [x_s.buffer, out_storage.buffer];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan softmax GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let w_s = weight
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan embedding: weight is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        // Upload indices to GPU buffer temp
        let idx_shape = Shape::new(vec![indices.len()]);
        let idx_storage = VulkanStorage::alloc_gpu(
            &idx_shape,
            DType {
                arith: ArithType::U32,
                storage: grim_tensor::dtype::Storage::Native,
            },
            ctx.device,
            ctx.physical_device,
        )?;
        let mut mapped_idx: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = vkMapMemory(
                ctx.device,
                idx_storage.memory,
                0,
                idx_storage.bytes as VkDeviceSize,
                0,
                &mut mapped_idx,
            );
            if res != VK_SUCCESS {
                return Err(Error::Backend(format!(
                    "vkMapMemory failed for indices buffer: {}",
                    res
                )));
            }
            std::ptr::copy_nonoverlapping(indices.as_ptr(), mapped_idx as *mut u32, indices.len());
            vkUnmapMemory(ctx.device, idx_storage.memory);
        }

        let w_dims = weight.shape().dims();
        let dim = w_dims[w_dims.len() - 1];
        let num_indices = indices.len();
        let size = num_indices * dim;

        let spirv_source: Vec<u8> = spirv_for(VulkanKernel::Embedding).to_vec();

        let buffers = [w_s.buffer, idx_storage.buffer, out_storage.buffer];
        let grid_x = size.div_ceil(256) as u32;

        let push = push_params(size as u32, dim as u32, 0, 0, 0, 0.0);

        run_compute_shader(ctx, &spirv_source, &buffers, grid_x, 1, 1, Some(&push))
            .map_err(|e| Error::Backend(format!("Vulkan embedding GPU dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
        let storage =
            VulkanStorage::alloc_gpu(shape, dtype.clone(), ctx.device, ctx.physical_device)?;

        let mut mapped: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            vkMapMemory(
                ctx.device,
                storage.memory,
                0,
                storage.bytes as VkDeviceSize,
                0,
                &mut mapped,
            )
        };
        if res != VK_SUCCESS {
            return Err(Error::Backend(format!(
                "vkMapMemory failed with status {}",
                res
            )));
        }

        unsafe {
            match dtype.arith {
                ArithType::BF16 => {
                    // Simulate BF16 precision via f32 round-trip while using FP32 kernels.
                    let dst = mapped as *mut f32;
                    for (i, &val) in data.iter().enumerate() {
                        *dst.add(i) = f32_to_bf16_to_f32(val);
                    }
                }
                _ => {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut f32, data.len());
                }
            }
            vkUnmapMemory(ctx.device, storage.memory);
        }

        Ok(Box::new(storage))
    }

    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        // Vulkan backend: MemAdvice is currently a no-op
        Ok(())
    }
}
