//! autograd_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::Result;
use grim_tensor::{AutogradOps, BackendStorage, CoreTensorOps, Shape};

use grim_backend_cpu::CpuDevice;

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl AutogradOps for MetalDevice {
    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let e_s = e.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal silu_mul_backward: e is not MetalStorage".into())
                })?;
                let g_s = g.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal silu_mul_backward: g is not MetalStorage".into())
                })?;
                let dw_s = dw.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal silu_mul_backward: dw is not MetalStorage".into())
                })?;
                let e_buf = e_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("e has no GPU buffer".into()))?;
                let g_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("g has no GPU buffer".into()))?;
                let dw_buf = dw_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("dw has no GPU buffer".into()))?;
                let df_storage = self.zeros(out_shape, DType::F32)?;
                let de_storage = self.zeros(out_shape, DType::F32)?;
                let df_buf = df_storage
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .unwrap()
                    .buffer
                    .as_ref()
                    .unwrap();
                let de_buf = de_storage
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .unwrap()
                    .buffer
                    .as_ref()
                    .unwrap();
                let cmd = self.get_or_create_command_buffer()?;
                let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder.setComputePipelineState(&inner.pipelines.silu_mul_backward);
                encoder.setBuffer_offset_atIndex(Some(e_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(g_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dw_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(df_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(de_buf), 0, 4);
                let total = out_shape.elem_count() as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &total as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }
                let threads = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total as usize + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                encoder.endEncoding();
                return Ok((
                    df_storage,
                    de_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }
        let cpu = CpuDevice::new();
        let e_cpu = e.to_cpu_vec_f32()?;
        let g_cpu = g.to_cpu_vec_f32()?;
        let dw_cpu = dw.to_cpu_vec_f32()?;
        let mut df = vec![0.0f32; out_shape.elem_count()];
        let mut de = vec![0.0f32; out_shape.elem_count()];
        for i in 0..df.len() {
            let s = 1.0 / (1.0 + (-e_cpu[i]).exp());
            df[i] = dw_cpu[i] * g_cpu[i] * s * (1.0 + e_cpu[i] * (1.0 - s));
            de[i] = dw_cpu[i] * s * e_cpu[i];
        }
        let df_storage = cpu.from_cpu(&df, out_shape, DType::F32)?;
        let de_storage = cpu.from_cpu(&de, out_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            return Ok((
                df_storage,
                de_storage,
                Box::new(MetalHandle { command_buffer }),
            ));
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((df_storage, de_storage, Box::new(MetalHandle)))
    }

    fn rmsnorm_backward(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        out_grad: &dyn BackendStorage,
        eps: f32,
        x_shape: &Shape,
        w_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rmsnorm_backward: x is not MetalStorage".into())
                })?;
                let w_s = weight
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("Metal rmsnorm_backward: weight is not MetalStorage".into())
                    })?;
                let g_s = out_grad
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend(
                            "Metal rmsnorm_backward: out_grad is not MetalStorage".into(),
                        )
                    })?;
                let x_buf = x_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("x has no GPU buffer".into()))?;
                let w_buf = w_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("weight has no GPU buffer".into()))?;
                let g_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("out_grad has no GPU buffer".into()))?;
                let row_len = w_shape.dims().last().copied().unwrap_or(1);
                let num_rows = x_shape
                    .dims()
                    .iter()
                    .take(x_shape.dims().len() - 1)
                    .product::<usize>()
                    / row_len.max(1);
                let total = x_shape.elem_count();
                let dx_storage = self.zeros(x_shape, DType::F32)?;
                let dx_s = dx_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let dx_buf = dx_s.buffer.as_ref().unwrap();
                let dw_storage = self.zeros(w_shape, DType::F32)?;
                let dw_s = dw_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let dw_buf = dw_s.buffer.as_ref().unwrap();
                let cmd = self.get_or_create_command_buffer()?;
                let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder.setComputePipelineState(&inner.pipelines.rmsnorm_backward);
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(g_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(dx_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(dw_buf), 0, 4);
                let row_len_val = row_len as i32;
                let eps_val = eps;
                let total_val = total as i32;
                let num_rows_val = num_rows as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &row_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                    encoder.setBytes_length_atIndex(
                        &eps_val as *const f32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        7,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_rows_val as *const i32 as *const std::ffi::c_void,
                        4,
                        8,
                    );
                }
                let threads = MTLSize::new(32, 1, 1);
                let groups = MTLSize::new((num_rows.max(1) + 31) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                encoder.endEncoding();
                return Ok((
                    dx_storage,
                    dw_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }
        let cpu = CpuDevice::new();
        let x_cpu = x.to_cpu_vec_f32()?;
        let w_cpu = weight.to_cpu_vec_f32()?;
        let g_cpu = out_grad.to_cpu_vec_f32()?;
        let row_len = w_shape.dims().last().copied().unwrap_or(1);
        let num_rows = x_shape.elem_count() / row_len.max(1);
        let mut dx = vec![0.0f32; x_shape.elem_count()];
        let mut dw = vec![0.0f32; w_shape.elem_count()];
        for r in 0..num_rows {
            let base = r * row_len;
            let mut ss = 0.0f32;
            let mut sum_gw = 0.0f32;
            for i in 0..row_len {
                let xv = x_cpu[base + i];
                let gv = g_cpu[base + i];
                let wv = w_cpu[i];
                ss += xv * xv;
                sum_gw += gv * wv * xv;
            }
            let rms = (ss / row_len.max(1) as f32 + eps).sqrt();
            let rms_inv = 1.0 / rms;
            let scale_sub = (sum_gw / row_len.max(1) as f32) * (rms_inv * rms_inv * rms_inv);
            for i in 0..row_len {
                let xv = x_cpu[base + i];
                let gv = g_cpu[base + i];
                let wv = w_cpu[i];
                dx[base + i] = (wv * rms_inv) * gv - xv * scale_sub;
                dw[i] += gv * (xv * rms_inv);
            }
        }
        let dx_storage = cpu.from_cpu(&dx, x_shape, DType::F32)?;
        let dw_storage = cpu.from_cpu(&dw, w_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            return Ok((
                dx_storage,
                dw_storage,
                Box::new(MetalHandle { command_buffer }),
            ));
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((dx_storage, dw_storage, Box::new(MetalHandle)))
    }

    fn rope_backward(
        &self,
        out_grad: &dyn BackendStorage,
        cos: &dyn BackendStorage,
        sin: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let g_s = out_grad
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend("Metal rope_backward: out_grad is not MetalStorage".into())
                    })?;
                let c_s = cos.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rope_backward: cos is not MetalStorage".into())
                })?;
                let s_s = sin.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("Metal rope_backward: sin is not MetalStorage".into())
                })?;
                let g_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("out_grad has no GPU buffer".into()))?;
                let c_buf = c_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("cos has no GPU buffer".into()))?;
                let s_buf = s_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("sin has no GPU buffer".into()))?;
                let half_dim = cos.shape().elem_count();
                let head_dim = half_dim * 2;
                let total_tokens = out_shape.elem_count() / head_dim.max(1);
                let total_pairs = (total_tokens * head_dim) / 2;
                let dx_storage = self.zeros(out_shape, DType::F32)?;
                let dx_s = dx_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let dx_buf = dx_s.buffer.as_ref().unwrap();
                let cmd = self.get_or_create_command_buffer()?;
                let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder.setComputePipelineState(&inner.pipelines.rope_backward);
                encoder.setBuffer_offset_atIndex(Some(g_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(c_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(s_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(dx_buf), 0, 3);
                let half_dim_val = half_dim as i32;
                let total_tokens_val = total_tokens as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &half_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_tokens_val as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }
                let threads = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total_pairs as usize + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                encoder.endEncoding();
                return Ok((
                    dx_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }
        let cpu = CpuDevice::new();
        let g_cpu = out_grad.to_cpu_vec_f32()?;
        let c_cpu = cos.to_cpu_vec_f32()?;
        let s_cpu = sin.to_cpu_vec_f32()?;
        let half_dim = cos.shape().elem_count();
        let head_dim = half_dim * 2;
        let total_tokens = out_shape.elem_count() / head_dim.max(1);
        let mut dx = vec![0.0f32; out_shape.elem_count()];
        for t in 0..total_tokens {
            let offset = t * head_dim;
            for i in 0..half_dim {
                let g0 = g_cpu[offset + i];
                let g1 = g_cpu[offset + half_dim + i];
                let c = c_cpu[i];
                let s = s_cpu[i];
                dx[offset + i] = g0 * c + g1 * s;
                dx[offset + half_dim + i] = -g0 * s + g1 * c;
            }
        }
        let dx_storage = cpu.from_cpu(&dx, out_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            return Ok((dx_storage, Box::new(MetalHandle { command_buffer })));
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((dx_storage, Box::new(MetalHandle)))
    }

    fn softmax_backward(
        &self,
        out_grad: &dyn BackendStorage,
        softmax_out: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let g_s = out_grad
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend(
                            "Metal softmax_backward: out_grad is not MetalStorage".into(),
                        )
                    })?;
                let s_s = softmax_out
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend(
                            "Metal softmax_backward: softmax_out is not MetalStorage".into(),
                        )
                    })?;
                let g_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("out_grad has no GPU buffer".into()))?;
                let s_buf = s_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("softmax_out has no GPU buffer".into()))?;
                let row_len = out_shape.dims().last().copied().unwrap_or(1);
                let total = out_shape.elem_count();
                let dx_storage = self.zeros(out_shape, DType::F32)?;
                let dx_s = dx_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let dx_buf = dx_s.buffer.as_ref().unwrap();
                let cmd = self.get_or_create_command_buffer()?;
                let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder.setComputePipelineState(&inner.pipelines.softmax_backward);
                encoder.setBuffer_offset_atIndex(Some(g_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(s_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dx_buf), 0, 2);
                let row_len_val = row_len as i32;
                let total_val = total as i32;
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &row_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder.setBytes_length_atIndex(
                        &total_val as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                }
                let threads = MTLSize::new(256, 1, 1);
                let groups = MTLSize::new(((total as usize + 255) / 256) as u64, 1, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                encoder.endEncoding();
                return Ok((
                    dx_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }
        let cpu = CpuDevice::new();
        let g_cpu = out_grad.to_cpu_vec_f32()?;
        let s_cpu = softmax_out.to_cpu_vec_f32()?;
        let row_len = out_shape.dims().last().copied().unwrap_or(1);
        let total = out_shape.elem_count();
        let mut dx = vec![0.0f32; total];
        let num_rows = total / row_len.max(1);
        for r in 0..num_rows {
            let base = r * row_len;
            let mut sum_g_s = 0.0f32;
            for j in 0..row_len {
                sum_g_s += g_cpu[base + j] * s_cpu[base + j];
            }
            for j in 0..row_len {
                dx[base + j] = s_cpu[base + j] * (g_cpu[base + j] - sum_g_s);
            }
        }
        let dx_storage = cpu.from_cpu(&dx, out_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            return Ok((dx_storage, Box::new(MetalHandle { command_buffer })));
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((dx_storage, Box::new(MetalHandle)))
    }

    #[allow(clippy::needless_range_loop)]
    fn embedding_backward(
        &self,
        out_grad: &dyn BackendStorage,
        token_ids: &[u32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                let g_s = out_grad
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| {
                        Error::Backend(
                            "Metal embedding_backward: out_grad is not MetalStorage".into(),
                        )
                    })?;
                let g_buf = g_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("out_grad has no GPU buffer".into()))?;
                let num_tokens = token_ids.len();
                if num_tokens == 0 || hidden_dim == 0 || vocab_size == 0 {
                    return Err(Error::Shape(
                        "embedding_backward: empty vocab/hidden/tokens".into(),
                    ));
                }
                let dw_shape = Shape::new(vec![vocab_size * hidden_dim]);
                let dw_storage = self.zeros(&dw_shape, DType::F32)?;
                let dw_s = dw_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let dw_buf = dw_s.buffer.as_ref().unwrap();
                // Upload token_ids as U32 buffer
                let ids_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
                let ids_storage =
                    self.from_cpu_bytes(&ids_bytes, &Shape::new(vec![num_tokens]), DType::U32)?;
                let ids_s = ids_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let ids_buf = ids_s.buffer.as_ref().unwrap();
                let cmd = self.get_or_create_command_buffer()?;
                // Zero-fill first
                let encoder0 = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder0.setComputePipelineState(&inner.pipelines.zeros_f32);
                encoder0.setBuffer_offset_atIndex(Some(dw_buf), 0, 0);
                let size_val = dw_shape.elem_count() as i32;
                unsafe {
                    encoder0.setBytes_length_atIndex(
                        &size_val as *const i32 as *const std::ffi::c_void,
                        4,
                        1,
                    );
                }
                let threads0 = MTLSize::new(256, 1, 1);
                let groups0 =
                    MTLSize::new(((dw_shape.elem_count() as usize + 255) / 256) as u64, 1, 1);
                encoder0.dispatchThreadgroups_threadsPerThreadgroup(groups0, threads0);
                encoder0.endEncoding();
                // Then scatter-add
                let encoder1 = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;
                encoder1.setComputePipelineState(&inner.pipelines.embedding_scatter_add);
                encoder1.setBuffer_offset_atIndex(Some(g_buf), 0, 0);
                encoder1.setBuffer_offset_atIndex(Some(ids_buf), 0, 1);
                encoder1.setBuffer_offset_atIndex(Some(dw_buf), 0, 2);
                let hidden_val = hidden_dim as i32;
                let num_tokens_val = num_tokens as i32;
                unsafe {
                    encoder1.setBytes_length_atIndex(
                        &hidden_val as *const i32 as *const std::ffi::c_void,
                        4,
                        3,
                    );
                    encoder1.setBytes_length_atIndex(
                        &num_tokens_val as *const i32 as *const std::ffi::c_void,
                        4,
                        4,
                    );
                }
                let total = num_tokens * hidden_dim;
                let threads1 = MTLSize::new(256, 1, 1);
                let groups1 = MTLSize::new(((total as usize + 255) / 256) as u64, 1, 1);
                encoder1.dispatchThreadgroups_threadsPerThreadgroup(groups1, threads1);
                encoder1.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                let dw_storage2 =
                    self.zeros(&Shape::new(vec![vocab_size, hidden_dim]), DType::F32)?;
                return Ok((
                    dw_storage2,
                    Box::new(MetalHandle {
                        command_buffer: cmd,
                    }),
                ));
            }
        }
        let cpu = CpuDevice::new();
        let g_cpu = out_grad.to_cpu_vec_f32()?;
        let num_tokens = token_ids.len();
        let dw_shape = Shape::new(vec![vocab_size, hidden_dim]);
        let mut dw = vec![0.0f32; dw_shape.elem_count()];
        for t in 0..num_tokens {
            let word_idx = token_ids[t] as usize;
            let base = t * hidden_dim;
            let dw_base = word_idx * hidden_dim;
            for h in 0..hidden_dim {
                dw[dw_base + h] += g_cpu[base + h];
            }
        }
        let dw_storage = cpu.from_cpu(&dw, &dw_shape, DType::F32)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            return Ok((dw_storage, Box::new(MetalHandle { command_buffer })));
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((dw_storage, Box::new(MetalHandle)))
    }
}
