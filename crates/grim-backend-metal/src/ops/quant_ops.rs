//! quant_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::{ ComputeHandle };
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage, CoreTensorOps, QuantOps, Shape };


#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl QuantOps for MetalDevice {
    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let (out, _handle) = self.quantize_on_device(x, format)?;
        Ok(out)
    }

    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        _format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // --- Apple Silicon GPU fast-path ---------------------------------------- Each thread computes one output element [row, col] by dequantizing its column of B on-the-fly inside the kernel.
        // Both A and B-packed must be device-resident MetalStorage buffers.
        #[cfg(target_vendor = "apple")]
        {
            let a_s = a.as_any().downcast_ref::<MetalStorage>();
            let b_s = b_packed.as_any().downcast_ref::<MetalStorage>();
            if let (Some(a_s), Some(b_s)) = (a_s, b_s) {
                if let (Some(a_buf), Some(b_buf)) = (a_s.buffer.as_ref(), b_s.buffer.as_ref()) {
                    if let Ok(ctx) = MetalContext::get() {
                        if let DTypeStorage::ResidualPacked(cfg) = b_packed.dtype().storage {
                            let residuals =
                                match b_packed.provenance() {
                                    QuantProvenance::WithResiduals {
                                        outlier_count,
                                        outlier_indices_offset,
                                        outlier_values_offset,
                                        backup1_bpw,
                                        backup1_codes_offset,
                                        backup1_scale_offset,
                                        backup2_bpw,
                                        backup2_codes_offset,
                                        backup2_scale_offset,
                                    } => (
                                        outlier_count,
                                        outlier_indices_offset,
                                        outlier_values_offset,
                                        backup1_bpw,
                                        backup1_codes_offset,
                                        backup1_scale_offset,
                                        backup2_bpw,
                                        backup2_codes_offset,
                                        backup2_scale_offset,
                                    ),
                                    _ => return Err(Error::Unimplemented(
                                        "Metal ResidualPacked requires WithResiduals provenance"
                                            .into(),
                                    )),
                                };
                            if cfg.bpw == 0 || cfg.bpw > 8 || k == 0 || n == 0 {
                                return Err(Error::Shape(
                                    "invalid ResidualPacked dimensions or bitwidth".into(),
                                ));
                            }
                            let row_stride = ((k * cfg.bpw as usize).div_ceil(8) + 255) / 256 * 256;
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    b_buf.contents() as *const u8,
                                    b_buf.length() as usize,
                                )
                            };
                            let decode_scales = |offset: usize| -> Vec<f32> {
                                if offset == 0 {
                                    vec![1.0; n]
                                } else {
                                    (0..n)
                                        .map(|i| {
                                            bytes.get(offset + i).copied().unwrap_or(255) as f32
                                                / 255.0
                                        })
                                        .collect()
                                }
                            };
                            let (
                                outlier_count,
                                oi_off,
                                ov_off,
                                b1,
                                b1_off,
                                b1_scale,
                                b2,
                                b2_off,
                                b2_scale,
                            ) = residuals;
                            let mut scales = vec![1.0f32; n];
                            scales[..b_scales.len().min(n)]
                                .copy_from_slice(&b_scales[..b_scales.len().min(n)]);
                            scales.extend(decode_scales(b1_scale));
                            scales.extend(decode_scales(b2_scale));
                            let mut indices = Vec::<u32>::with_capacity(outlier_count);
                            let mut values = Vec::<f32>::with_capacity(outlier_count);
                            for i in 0..outlier_count {
                                let p = oi_off + i * 6;
                                let q = ov_off + i * 6;
                                if p + 4 > bytes.len() || q + 2 > bytes.len() {
                                    return Err(Error::Backend(
                                        "ResidualPacked outlier region exceeds Metal buffer".into(),
                                    ));
                                }
                                indices
                                    .push(u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()));
                                let h = u16::from_le_bytes(bytes[q..q + 2].try_into().unwrap());
                                let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
                                let exp = ((h >> 10) & 0x1f) as i32;
                                let mant = (h & 0x3ff) as u32;
                                values.push(if exp == 0 {
                                    sign * (mant as f32) * 2.0f32.powi(-24)
                                } else {
                                    sign * (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 25)
                                });
                            }
                            let make_buf = |ptr: *const std::ffi::c_void, len: usize| {
                                ctx.device
                                    .newBufferWithBytes_length_options(
                                        ptr,
                                        len as u64,
                                        MTLResourceOptions::StorageModeShared,
                                    )
                                    .ok()
                                    .ok_or_else(|| {
                                        Error::from(MetalError::AllocationFailed(
                                            "ResidualPacked auxiliary buffer allocation failed"
                                                .into(),
                                        ))
                                    })
                            };
                            let scales_buf =
                                make_buf(scales.as_ptr() as *const _, scales.len() * 4)?;
                            let idx_buf =
                                make_buf(indices.as_ptr() as *const _, indices.len().max(1) * 4)?;
                            let val_buf =
                                make_buf(values.as_ptr() as *const _, values.len().max(1) * 4)?;
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_buf = out_storage
                                .as_any()
                                .downcast_ref::<MetalStorage>()
                                .unwrap()
                                .buffer
                                .as_ref()
                                .unwrap();
                            let cmd = self.get_or_create_command_buffer()?;
                            let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;
                            enc.setComputePipelineState(&ctx.pipelines.residualpacked_matmul);
                            for (buf, idx) in [
                                Some(a_buf),
                                Some(b_buf),
                                Some(&scales_buf),
                                Some(&idx_buf),
                                Some(&val_buf),
                                Some(out_buf),
                            ]
                            .iter()
                            .enumerate()
                            {
                                enc.setBuffer_offset_atIndex(*buf, 0, idx as usize);
                            }
                            let vals = [
                                m as i32,
                                n as i32,
                                k as i32,
                                cfg.bpw as i32,
                                row_stride as i32,
                                0,
                                b1 as i32,
                                b1_off as i32,
                                b1_scale as i32,
                                b2 as i32,
                                b2_off as i32,
                                b2_scale as i32,
                                outlier_count as i32,
                            ];
                            unsafe {
                                for (i, v) in vals.iter().enumerate() {
                                    enc.setBytes_length_atIndex(
                                        v as *const i32 as *const _,
                                        4,
                                        6 + i,
                                    );
                                }
                            }
                            enc.dispatchThreadgroups(
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1),
                                MTLSize::new(16, 16, 1),
                            );
                            enc.endEncoding();
                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd,
                                }),
                            ));
                        }
                        // Q4_K fast-path
                        if let DTypeStorage::KQuant(KQuantScheme::Q4K) = b_packed.dtype().storage {
                            if k >= 256 && k % 256 == 0 {
                                let out_storage = self.zeros(out_shape, DType::F32)?;
                                let out_s =
                                    out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                                let out_buf = out_s.buffer.as_ref().unwrap();

                                let cmd_buffer = self.get_or_create_command_buffer()?;
                                let encoder =
                                    cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                        Error::from(MetalError::Ffi(
                                            "Failed to create compute encoder".into(),
                                        ))
                                    })?;

                                encoder
                                    .setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_q4k);
                                encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                                let m_val = m as i32;
                                let n_val = n as i32;
                                let k_val = k as i32;
                                unsafe {
                                    encoder.setBytes_length_atIndex(
                                        &m_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        3,
                                    );
                                    encoder.setBytes_length_atIndex(
                                        &n_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        4,
                                    );
                                    encoder.setBytes_length_atIndex(
                                        &k_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        5,
                                    );
                                }

                                let threads_per_group = MTLSize::new(16, 16, 1);
                                let groups =
                                    MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                    groups,
                                    threads_per_group,
                                );
                                encoder.endEncoding();

                                return Ok((
                                    out_storage,
                                    Box::new(MetalHandle {
                                        command_buffer: cmd_buffer,
                                    }),
                                ));
                            }
                        }

                        // FP8 fast-path
                        if let DTypeStorage::FloatPack(FloatPackScheme::Fp8) =
                            b_packed.dtype().storage
                        {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_fp8);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }

                            let threads_per_group = MTLSize::new(16, 16, 1);
                            let groups =
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                groups,
                                threads_per_group,
                            );
                            encoder.endEncoding();

                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd_buffer,
                                }),
                            ));
                        }

                        // MXFP4 fast-path — fused dequant+INT8 GEMM via simdgroup_matrix
                        if let DTypeStorage::FloatPack(FloatPackScheme::MxFp4) =
                            b_packed.dtype().storage
                        {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder
                                .setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_mxfp4);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    6,
                                );
                            }

                            // 16×16 threadgroup = 256 threads, matching CUDA block size.
                            let threads_per_group = MTLSize::new(16, 16, 1);
                            let groups =
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                groups,
                                threads_per_group,
                            );
                            encoder.endEncoding();

                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd_buffer,
                                }),
                            ));
                        }

                        // Q8_0 scaled-quantized fallback (k >= 32, k % 32 == 0)
                        if k >= 32 && k % 32 == 0 {
                            // Pad / truncate scales to exactly n * (k/32) entries.
                            let blocks_per_col = k / 32;
                            let scales_len = n * blocks_per_col;
                            let mut scales_f32 = vec![1.0f32; scales_len];
                            let copy_len = b_scales.len().min(scales_len);
                            scales_f32[..copy_len].copy_from_slice(&b_scales[..copy_len]);

                            let scales_buf = ctx
                                .device
                                .newBufferWithBytes_length_options(
                                    scales_f32.as_ptr() as *const std::ffi::c_void,
                                    (scales_f32.len() * 4) as u64,
                                    MTLResourceOptions::StorageModeShared,
                                )
                                .ok_or_else(|| {
                                    Error::from(MetalError::AllocationFailed(
                                        "Failed to allocate scales buffer".into(),
                                    ))
                                })?;

                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&ctx.pipelines.quantized_matmul);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(&scales_buf), 0, 2);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);

                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    6,
                                );
                            }

                            // 16×16 threadgroup = 256 threads, matching CUDA block size.
                            let threads_per_group = MTLSize::new(16, 16, 1);
                            let groups =
                                MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                groups,
                                threads_per_group,
                            );
                            encoder.endEncoding();

                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd_buffer,
                                }),
                            ));
                        // IQ2/IQ3 fused dequant + GEMM fast-paths
                        if let DTypeStorage::KQuant(KQuantScheme::IQ2XXS) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq2xxs);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ2XS) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq2xs);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ2S) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq2s);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ3XXS) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq3xxs);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ3S) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq3s);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    3,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    4,
                                );
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }


                        // IQ2/IQ3 fused dequant + GEMM fast-paths
                        if let DTypeStorage::KQuant(KQuantScheme::IQ2XXS) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq2xxs);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ2XS) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq2xs);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ2S) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq2s);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ3XXS) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq3xxs);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }

                        if let DTypeStorage::KQuant(KQuantScheme::IQ3S) = b_packed.dtype().storage {
                            let out_storage = self.zeros(out_shape, DType::F32)?;
                            let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();
                            let cmd_buffer = self.get_or_create_command_buffer()?;
                            let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                            })?;
                            encoder.setComputePipelineState(&ctx.pipelines.fused_dequant_gemm_iq3s);
                            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                            let m_val = m as i32;
                            let n_val = n as i32;
                            let k_val = k as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void, 4, 3);
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void, 4, 4);
                                encoder.setBytes_length_atIndex(
                                    &k_val as *const i32 as *const std::ffi::c_void, 4, 5);
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let groups = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            encoder.endEncoding();
                            return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd_buffer })));
                        }


                        }
                    }
                }
            }
        }
        // --- end GPU fast-path ---------------------------------------------------

        // CPU fallback: dequant b and compute matmul on host.
        tracing::warn!("Metal quantized_matmul: falling back to CPU execution");
        let a_vec = a.to_cpu_vec_f32()?;
        let mut b_dequant = vec![0.0f32; k * n];
        let blocks_per_col = k / 32;

        #[cfg(target_vendor = "apple")]
        let b_bytes = if let Some(m_s) = b_packed.as_any().downcast_ref::<MetalStorage>() {
            if let Some(ref buf) = m_s.buffer {
                let ptr = buf.contents() as *const u8;
                let len = m_s.shape.elem_count();
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            } else if let Some(ref d) = m_s.data {
                d.lock().unwrap_or_else(|e| e.into_inner()).clone()
            } else {
                vec![0u8; k * n]
            }
        } else {
            vec![0u8; k * n]
        };

        #[cfg(not(target_vendor = "apple"))]
        let b_bytes = if let Some(m_s) = b_packed.as_any().downcast_ref::<MetalStorage>() {
            m_s.data.lock().unwrap_or_else(|e| e.into_inner()).clone()
        } else {
            vec![0u8; k * n]
        };

        for col in 0..n {
            for block in 0..blocks_per_col {
                let scale_idx = col * blocks_per_col + block;
                let scale = if scale_idx < b_scales.len() {
                    b_scales[scale_idx]
                } else {
                    1.0f32
                };
                for i in 0..32 {
                    let byte_offset = (col * blocks_per_col + block) * 32 + i;
                    let byte_val = if byte_offset < b_bytes.len() {
                        b_bytes[byte_offset]
                    } else {
                        128u8
                    };
                    let q_val = (byte_val as i16 - 128) as f32 / 127.0f32;
                    let r = block * 32 + i;
                    if r < k {
                        b_dequant[r * n + col] = q_val * scale;
                    }
                }
            }
        }

        let mut c_vec = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_vec[row * k + p] * b_dequant[p * n + col];
                }
                c_vec[row * n + col] = sum;
            }
        }

        let out_storage = self.from_cpu(&c_vec, out_shape, a.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        residuals: Option<&grim_tensor::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Basic validation - only Q8_0 (8-bit) supported, requires k >= 32 and block-aligned
        if default_bpw != 8 || k < 32 || k % 32 != 0 {
            return Err(Error::Unimplemented(
                "Metal Q8_0 backward supports only 8-bit block-aligned tensors".into(),
            ));
        }

        // For residuals (outliers, backup layers), fall back to CPU
        // This matches ROCm behavior where residuals cause a fallback path
        if let Some(res) = residuals {
            if res.outlier_count > 0 || res.backup1_bpw > 0 || res.backup2_bpw > 0 {
                let dy_vec = dy.to_cpu_vec_f32()?;
                let b_bytes = b_packed.to_cpu_vec_f32()?;
                let mut dx = vec![0.0f32; m * k];
                let blocks_per_col = k / 32;

                for row in 0..m {
                    for ki in 0..k {
                        let block = ki / 32;
                        let in_block = ki % 32;
                        let mut sum = 0.0f32;
                        for col in 0..n {
                            let idx = (col * blocks_per_col + block) * 32 + in_block;
                            let q = b_bytes.get(idx).copied().unwrap_or(0.0);
                            let scale = b_scales
                                .get(col * blocks_per_col + block)
                                .copied()
                                .unwrap_or(1.0);
                            sum += dy_vec[row * n + col] * q * scale;
                        }
                        dx[row * k + ki] = sum;
                    }
                }
                return Ok((
                    self.from_cpu(&dx, out_shape, DType::F32)?,
                    Box::new(MetalHandle),
                ));
            }
        }

        // Apple Metal GPU fast-path
        #[cfg(target_vendor = "apple")]
        if let Some(ref inner) = self.inner {
            let dy_s = dy.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                Error::Backend("Metal Q8_0 backward dy is not MetalStorage".into())
            })?;
            let b_s = b_packed
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| {
                    Error::Backend("Metal Q8_0 backward b is not MetalStorage".into())
                })?;

            let dy_buf = dy_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("Metal Q8_0 backward dy has no GPU buffer".into()))?;
            let b_buf = b_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("Metal Q8_0 backward b has no GPU buffer".into()))?;

            let ctx = MetalContext::get()?;
            let scale_count = n * (k / 32);
            let mut scales = vec![1.0f32; scale_count];
            let copy_len = b_scales.len().min(scale_count);
            scales[..copy_len].copy_from_slice(&b_scales[..copy_len]);

            let scales_buf = ctx
                .device
                .newBufferWithBytes_length_options(
                    scales.as_ptr() as *const std::ffi::c_void,
                    (scales.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    Error::from(MetalError::Ffi(
                        "Failed to allocate Q8_0 scale buffer".into(),
                    ))
                })?;

            let dx_storage = self.zeros(out_shape, DType::F32)?;
            let dx_s = dx_storage
                .as_any()
                .downcast_ref::<MetalStorage>()
                .ok_or_else(|| Error::Backend("dx_storage is not MetalStorage".into()))?;
            let dx_buf = dx_s
                .buffer
                .as_ref()
                .ok_or_else(|| Error::Backend("dx storage has no GPU buffer".into()))?;

            let cmd = self.get_or_create_command_buffer()?;
            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
            })?;

            encoder.setComputePipelineState(&inner.pipelines.quantized_matmul_backward);
            encoder.setBuffer_offset_atIndex(Some(dy_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&scales_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(dx_buf), 0, 3);

            let m_i = m as i32;
            let n_i = n as i32;
            let k_i = k as i32;
            unsafe {
                encoder.setBytes_length_atIndex(
                    &m_i as *const i32 as *const std::ffi::c_void,
                    4,
                    4,
                );
                encoder.setBytes_length_atIndex(
                    &n_i as *const i32 as *const std::ffi::c_void,
                    4,
                    5,
                );
                encoder.setBytes_length_atIndex(
                    &k_i as *const i32 as *const std::ffi::c_void,
                    4,
                    6,
                );
            }

            // Grid: (k/16) threadgroups in x, (m/16) in y, 1 in z
            // Each thread Computes dx[row, k_idx] for row < m, k_idx < k
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize::new(((k + 15) / 16) as u64, ((m + 15) / 16) as u64, 1),
                MTLSize::new(16, 16, 1),
            );
            encoder.endEncoding();

            return Ok((
                dx_storage,
                Box::new(MetalHandle {
                    command_buffer: cmd,
                }),
            ));
        }

        // Non-Apple fallback to CPU
        // CPU fallback implementation for quantized matmul backward
        let dy_vec = dy.to_cpu_vec_f32()?;
        let b_bytes = b_packed.to_cpu_vec_f32()?;
        let mut dx = vec![0.0f32; m * k];
        let blocks_per_col = k / 32;

        for row in 0..m {
            for ki in 0..k {
                let block = ki / 32;
                let in_block = ki % 32;
                let mut sum = 0.0f32;
                for col in 0..n {
                    let idx = (col * blocks_per_col + block) * 32 + in_block;
                    let q = b_bytes.get(idx).copied().unwrap_or(0.0);
                    let scale = b_scales
                        .get(col * blocks_per_col + block)
                        .copied()
                        .unwrap_or(1.0);
                    sum += dy_vec[row * n + col] * q * scale;
                }
                dx[row * k + ki] = sum;
            }
        }
        Ok((
            self.from_cpu(&dx, out_shape, DType::F32)?,
            Box::new(MetalHandle),
        ))
    }
}
