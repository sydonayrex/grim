//! `QuantOps` implementation for VulkanDevice.
//! Extracted from lib.rs (modularization): trait impls live in `device/`, dispatch plumbing in `kernel.rs`, buffers in.

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{
    DType, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, BackendStorage, CoreTensorOps, QuantOps, Shape};

use crate::context::global_context;
use crate::kernel::{
    VulkanKernel, push_params, push_params_backward, run_compute_shader, run_compute_shader_kernel,
    spirv_for,
};
use crate::{VulkanDevice, VulkanHandle, VulkanStorage, extract_raw_bytes};
use grim_tensor::MemoryOps;

impl QuantOps for VulkanDevice {
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

        // Try GPU fused dequant dispatch if both inputs are VulkanStorage.
        // Kernel selection is based on the weight tensor's actual dtype - NOT on k %.
        if let (Some(a_s), Some(b_s)) = (
            a.as_any().downcast_ref::<VulkanStorage>(),
            b_packed.as_any().downcast_ref::<VulkanStorage>(),
        ) {
            use grim_tensor::dtype::{FloatPackScheme, KQuantScheme, Storage};
            // Map the weight dtype to the kernel that knows its block layout.
            // Formats not handled by any Vulkan fused kernel skip GPU dispatch and fall through to.
            let b_weight_dtype = b_packed.dtype();
            let maybe_kernel = match &b_weight_dtype.storage {
                Storage::KQuant(KQuantScheme::Q4K) => Some(VulkanKernel::FusedDequantGemmQ4K),
                Storage::KQuant(KQuantScheme::Q5K) => Some(VulkanKernel::FusedDequantGemmQ5K),
                Storage::KQuant(KQuantScheme::Q6K) => Some(VulkanKernel::FusedDequantGemmQ6K),
                Storage::KQuant(KQuantScheme::Q80) => Some(VulkanKernel::FusedDequantGemmQ80),
                Storage::KQuant(KQuantScheme::IQ4NL) => Some(VulkanKernel::FusedDequantGemmIQ4NL),
                Storage::KQuant(KQuantScheme::IQ4XS) => Some(VulkanKernel::FusedDequantGemmIQ4XS),
                Storage::KQuant(KQuantScheme::IQ3XXS) => Some(VulkanKernel::FusedDequantGemmIQ3XXS),
                Storage::KQuant(KQuantScheme::IQ3S) => Some(VulkanKernel::FusedDequantGemmIQ3S),
                Storage::KQuant(KQuantScheme::IQ2XXS) => Some(VulkanKernel::FusedDequantGemmIQ2XXS),
                Storage::KQuant(KQuantScheme::IQ2XS) => Some(VulkanKernel::FusedDequantGemmIQ2XS),
                Storage::KQuant(KQuantScheme::IQ2S) => Some(VulkanKernel::FusedDequantGemmIQ2S),
                Storage::FloatPack(FloatPackScheme::Fp8) => {
                    // T1 caps gate: without FP8 shader support, fall through to the CPU path
                    // rather than dispatch the FP8 fused-dequant shader.
                    if self
                        .caps
                        .supports_quant_format(grim_tensor::QuantFormat::Fp8)
                    {
                        Some(VulkanKernel::FusedDequantGemmFp8E4M3)
                    } else {
                        None
                    }
                }
                Storage::FloatPack(FloatPackScheme::MxFp4) => {
                    Some(VulkanKernel::FusedDequantGemmMxFp4)
                }
                Storage::FloatPack(FloatPackScheme::NvFp4) => {
                    Some(VulkanKernel::FusedDequantGemmNvFp4)
                }
                Storage::CompressedTensorsW8A8Fp8 => Some(VulkanKernel::FusedDequantGemmW8A8Fp8),
                Storage::CompressedTensorsW8A8Int8 => Some(VulkanKernel::FusedDequantGemmW8A8Int8),
                Storage::GroupInt(_) | Storage::W4A16(_) => Some(VulkanKernel::MarlinGemm),
                other => {
                    tracing::warn!(
                        "Vulkan quantized_matmul: no GPU kernel for dtype storage {:?}; \
                         falling back to CPU",
                        other
                    );
                    None
                }
            };
            if let Some(kernel) = maybe_kernel {
                let ctx_guard = global_context();
                if let Some(ctx) = ctx_guard.as_ref() {
                    if let Ok(out_storage) = VulkanStorage::alloc_device_local_gpu(
                        out_shape,
                        DType::F32,
                        ctx.device,
                        ctx.physical_device,
                    ) {
                        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
                        let grid_x = n.div_ceil(16) as u32;
                        let grid_y = m.div_ceil(16) as u32;
                        let push = push_params(0, 0, k as u32, n as u32, m as u32, 0.0);

                        match run_compute_shader_kernel(
                            ctx,
                            kernel,
                            &buffers,
                            grid_x,
                            grid_y,
                            1,
                            Some(&push),
                        ) {
                            Ok(()) => {
                                return Ok((Box::new(out_storage), Box::new(VulkanHandle)));
                            }
                            // Surface the real Vulkan error instead of silently dropping it;
                            // binding-count mismatches become Err here (P0-1 guard).
                            Err(e) => tracing::warn!(
                                "Vulkan quantized_matmul GPU dispatch failed ({e:?}); falling back to CPU"
                            ),
                        }
                    }
                }
            }
        }

        tracing::warn!("Vulkan quantized_matmul: falling back to CPU execution");
        let a_vec = a.to_cpu_vec_f32()?;
        let mut b_dequant = vec![0.0f32; k * n];
        let blocks_per_col = k / 32;

        // Safety contract for the CPU fallback dequant loop below: the loop decodes bytes as Q8_0 (signed int8, block size 32, scale from b_scales).
        // Calling to_cpu_vec_f32() on a packed quantized buffer reinterprets the raw packed bytes as f32 -.
        use grim_tensor::dtype::{BlockDtype, FloatPackScheme, KQuantScheme, Storage};
        let b_weight_dtype = b_packed.dtype();

        // Use grim_quant's dequant functions for formats that have them;
        // Q8_0 falls through to the legacy inline Q8_0 decoder below.
        let grim_dequant: Option<Vec<f32>> = match &b_weight_dtype.storage {
            Storage::KQuant(scheme) => {
                let b_bytes_cpu: Vec<u8> = extract_raw_bytes(b_packed)?;
                Some(match scheme {
                    KQuantScheme::Q4K => grim_quant::dequant_q4k(&b_bytes_cpu, k * n)?,
                    KQuantScheme::Q5K => grim_quant::dequant_q5k(&b_bytes_cpu, k * n)?,
                    KQuantScheme::Q6K => grim_quant::dequant_q6k(&b_bytes_cpu, k * n)?,
                    KQuantScheme::Q80 => grim_quant::dequant_q80(&b_bytes_cpu, k * n)?,
                    KQuantScheme::Q2K => grim_quant::dequant_q2k(&b_bytes_cpu, k * n)?,
                    KQuantScheme::Q3K => grim_quant::dequant_q3k(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ4XS => grim_quant::dequant_iq4xs(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ3XXS => grim_quant::dequant_iq3xxs(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ3S => grim_quant::dequant_iq3s(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ2XXS => grim_quant::dequant_iq2xxs(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ2XS => grim_quant::dequant_iq2xs(&b_bytes_cpu, k * n)?,
                    KQuantScheme::IQ2S => grim_quant::dequant_iq2s(&b_bytes_cpu, k * n)?,
                })
            }
            Storage::FloatPack(scheme) => {
                let b_bytes_cpu: Vec<u8> = extract_raw_bytes(b_packed)?;
                Some(match scheme {
                    FloatPackScheme::Fp4 => grim_quant::dequant_fp4(&b_bytes_cpu, k * n)?,
                    FloatPackScheme::Nf4 => grim_quant::dequant_nf4(&b_bytes_cpu, k * n)?,
                    FloatPackScheme::Fp8 => grim_quant::dequant_fp8(&b_bytes_cpu, k * n)?,
                    FloatPackScheme::MxFp4 => grim_quant::dequant_mxfp4(&b_bytes_cpu, k * n)?,
                    FloatPackScheme::MxFp8 => grim_quant::dequant_mxfp8(&b_bytes_cpu, k * n)?,
                    FloatPackScheme::NvFp4 => grim_quant::dequant_nvfp4(&b_bytes_cpu, k * n)?,
                })
            }
            Storage::Block(dtype) => {
                let b_bytes_cpu: Vec<u8> = extract_raw_bytes(b_packed)?;
                Some(match dtype {
                    BlockDtype::Fp4 => grim_quant::dequant_fp4_block16(&b_bytes_cpu, k * n)?,
                    BlockDtype::Nf4 => {
                        // NF4 block-16 shares the fp4 dequant path in grim_quant.
                        grim_quant::dequant_fp4_block16(&b_bytes_cpu, k * n)?
                    }
                    BlockDtype::Fp8 => grim_quant::dequant_fp8_block16(&b_bytes_cpu, k * n)?,
                    BlockDtype::Fp4Block16 => grim_quant::dequant_fp4_block16(&b_bytes_cpu, k * n)?,
                    BlockDtype::Fp8Block16 => grim_quant::dequant_fp8_block16(&b_bytes_cpu, k * n)?,
                })
            }
            Storage::CompressedTensorsW8A8Fp8 => {
                let b_bytes_cpu: Vec<u8> = extract_raw_bytes(b_packed)?;
                // [u64 scale_len][scales F32][FP8 codes]
                let scale_len = u64::from_le_bytes(b_bytes_cpu[..8].try_into().unwrap()) as usize;
                let scales = &b_bytes_cpu[8..8 + scale_len];
                let codes = &b_bytes_cpu[8 + scale_len..];
                let mut out = Vec::with_capacity(k * n);
                for col in 0..n {
                    let scale = if scale_len >= n * 4 {
                        f32::from_le_bytes(scales[col * 4..col * 4 + 4].try_into().unwrap())
                    } else {
                        f32::from_le_bytes(scales[..4].try_into().unwrap())
                    };
                    for r in 0..k {
                        let code = codes[col * k + r];
                        out.push(grim_quant::fp8_e4m3_to_f32(code) * scale);
                    }
                }
                Some(out)
            }
            Storage::CompressedTensorsW8A8Int8 => {
                let b_bytes_cpu: Vec<u8> = extract_raw_bytes(b_packed)?;
                let scale_len = u64::from_le_bytes(b_bytes_cpu[..8].try_into().unwrap()) as usize;
                let scales = &b_bytes_cpu[8..8 + scale_len];
                let codes = &b_bytes_cpu[8 + scale_len..];
                let mut out = Vec::with_capacity(k * n);
                for col in 0..n {
                    let scale = if scale_len >= n * 4 {
                        f32::from_le_bytes(scales[col * 4..col * 4 + 4].try_into().unwrap())
                    } else {
                        f32::from_le_bytes(scales[..4].try_into().unwrap())
                    };
                    for r in 0..k {
                        let code = codes[col * k + r] as i8;
                        out.push((code as f32) * scale);
                    }
                }
                Some(out)
            }
            Storage::GroupInt(cfg) => {
                let b_bytes_cpu: Vec<u8> = extract_raw_bytes(b_packed)?;
                Some(grim_quant::dequant_gptq_group_int(
                    &b_bytes_cpu,
                    &[],
                    &[],
                    None,
                    &[k, n],
                    cfg.bits as u32,
                    cfg.group_size,
                )?)
            }
            _ => None, // ResidualPacked, Native — handled below.
        };

        if let Some(dequantized) = grim_dequant {
            // Copy dequantized B into b_dequant.
            b_dequant.copy_from_slice(&dequantized);
        } else if matches!(b_weight_dtype.storage, Storage::KQuant(KQuantScheme::Q80)) {
            // Legacy Q8_0 inline decoder (kept for backward compatibility).
            // Extract raw bytes from the Vulkan buffer without reinterpreting as f32.
            let b_bytes = extract_raw_bytes(b_packed)?;
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
        } else {
            return Err(Error::Backend(format!(
                "Vulkan quantized_matmul CPU fallback does not support weight dtype {:?}; \
                 use the GPU path or a CPU backend.",
                b_weight_dtype.storage
            )));
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
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let (out, _handle) = self.quantize_on_device(x, format)?;
        Ok(out)
    }

    fn fused_quant_gemm(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        format: QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        let a_s = a.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan fused_quant_gemm: a is not VulkanStorage".into())
        })?;
        let b_s = b.as_any().downcast_ref::<VulkanStorage>().ok_or_else(|| {
            Error::Backend("Vulkan fused_quant_gemm: b is not VulkanStorage".into())
        })?;

        let kernel = match format {
            QuantFormat::Q8_0 => VulkanKernel::FusedQuantGemmQ80,
            QuantFormat::Fp8 => VulkanKernel::FusedQuantGemmFp8,
            other => {
                return Err(Error::Backend(format!(
                    "Vulkan fused_quant_gemm: unsupported format {:?}",
                    other
                )));
            }
        };

        let (ctx_device, ctx_physical_device) = {
            let ctx_guard = global_context();
            let ctx = ctx_guard
                .as_ref()
                .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
            (ctx.device, ctx.physical_device)
        };

        let out_storage =
            VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx_device, ctx_physical_device)?;
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let buffers = [a_s.buffer, b_s.buffer, out_storage.buffer];
        let grid_x = n.div_ceil(16) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let push = push_params(0, 0, k as u32, n as u32, m as u32, 0.0);

        run_compute_shader_kernel(ctx, kernel, &buffers, grid_x, grid_y, 1, Some(&push))?;
        Ok((Box::new(out_storage), Box::new(VulkanHandle)))
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
        let dy_s = dy
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan backward dx dy is not VulkanStorage".into()))?;
        let b_s = b_packed
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan backward dx b_packed is not VulkanStorage".into())
            })?;

        // Extract context device/physical_device pointers without holding the lock - from_cpu_bytes
        // also locks GLOBAL_CONTEXT, so we must release here to avoid deadlock.
        let (ctx_device, ctx_physical_device) = {
            let ctx_guard = global_context();
            let ctx = ctx_guard
                .as_ref()
                .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;
            (ctx.device, ctx.physical_device)
        };

        // Allocate dX output [M, K] f32.
        let dx = VulkanStorage::alloc_gpu(out_shape, DType::F32, ctx_device, ctx_physical_device)?;

        // --- Extract residual / outlier metadata from the residuals handle ---
        let outlier_count = residuals.map(|r| r.outlier_count).unwrap_or(0);
        let backup1_bpw = residuals.map(|r| r.backup1_bpw).unwrap_or(0);
        let backup1_codes_offset = residuals.map(|r| r.backup1_codes_offset).unwrap_or(0);
        let backup1_scale_offset = residuals.map(|r| r.backup1_scale_offset).unwrap_or(0);
        let backup2_bpw = residuals.map(|r| r.backup2_bpw).unwrap_or(0);
        let backup2_codes_offset = residuals.map(|r| r.backup2_codes_offset).unwrap_or(0);
        let backup2_scale_offset = residuals.map(|r| r.backup2_scale_offset).unwrap_or(0);

        // --- Extract outlier index/value data from the tensor's provenance --- `QuantizedMatmulBackwardResiduals::from_tensor` leaves
        // the raw device pointers null; the actual host-decoded outlier vectors live in `QuantProvenance::WithResiduals`.
        let prov = b_s.provenance();
        let (outlier_indices_host, outlier_values_host) = match &prov {
            QuantProvenance::WithResiduals {
                outlier_indices,
                outlier_values_bits,
                ..
            } => {
                let indices: Vec<u8> = outlier_indices
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                let values: Vec<u8> = outlier_values_bits
                    .iter()
                    .flat_map(|v| f32::from_bits(*v).to_ne_bytes())
                    .collect();
                (indices, values)
            }
            _ => (Vec::new(), Vec::new()),
        };

        // --- Upload outlier buffers (binding 3 = indices u32, binding 4 = values f32) --- When
        // outlier_count == 0 the shader checks the count before accessing these buffers, so minimal dummies suffice.
        let (outlier_idx_box, outlier_val_box) = if outlier_count > 0
            && !outlier_indices_host.is_empty()
            && !outlier_values_host.is_empty()
        {
            let idx = self.from_cpu_bytes(
                &outlier_indices_host,
                &Shape::from_slice(&[outlier_indices_host.len()]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
            )?;
            let val = self.from_cpu_bytes(
                &outlier_values_host,
                &Shape::from_slice(&[outlier_values_host.len()]),
                DType::F32,
            )?;
            (idx, val)
        } else {
            let dummy = [0u8; 1];
            let idx = self.from_cpu_bytes(
                &dummy,
                &Shape::from_slice(&[1]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?;
            let val = self.from_cpu_bytes(
                &dummy,
                &Shape::from_slice(&[1]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?;
            (idx, val)
        };

        let outlier_idx_s = outlier_idx_box
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("outlier indices storage is not VulkanStorage".into()))?;
        let outlier_val_s = outlier_val_box
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("outlier values storage is not VulkanStorage".into()))?;

        // --- Select shader variant based on the weight's storage format ---
        let (kernel, has_scales) = match b_s.dtype().storage {
            DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                (VulkanKernel::QuantizedMatmulBackwardDx, false)
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                (VulkanKernel::QuantizedMatmulBackwardDxQ8_0, false)
            }
            // ResidualPacked, all other KQuant (Q5K, Q6K, IQ*), FloatPack,
            // Block, GroupInt — use the generic unpack_weight path.
            _ => (
                VulkanKernel::QuantizedMatmulBackwardDxGeneric,
                !b_scales.is_empty(),
            ),
        };

        // --- Upload per-column f32 scales for the generic shader (binding 5) ---
        let scales_storage_box = if has_scales {
            let f32_scale_bytes: Vec<u8> = b_scales.iter().flat_map(|&s| s.to_le_bytes()).collect();
            Some(self.from_cpu_bytes(
                &f32_scale_bytes,
                &Shape::from_slice(&[b_scales.len() * 4]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?)
        } else {
            // Dummy buffer so binding 5 always has a valid Vulkan buffer handle.
            let dummy = [0u8; 1];
            Some(self.from_cpu_bytes(
                &dummy,
                &Shape::from_slice(&[1]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
            )?)
        };
        let scales_s = scales_storage_box
            .as_ref()
            .and_then(|s| s.as_any().downcast_ref::<VulkanStorage>())
            .ok_or_else(|| Error::Backend("scales storage is not VulkanStorage".into()))?;

        // --- Build extended push constants ---
        let push = push_params_backward(
            k as u32,
            n as u32,
            m as u32,
            default_bpw as u32,
            outlier_count as u32,
            backup1_bpw as u32,
            backup1_codes_offset as u32,
            backup1_scale_offset as u32,
            backup2_bpw as u32,
            backup2_codes_offset as u32,
            backup2_scale_offset as u32,
            has_scales,
            1.0, // grad_scale = 1.0 for STE identity (straight-through estimator)
        );

        // --- Build GPU buffer binding list --- bindings:
        // [0]=dY, [1]=B_codes, [2]=dX, [3]=outlier_indices, [4]=outlier_values, [5]=scales_u8 (generic only)
        let mut buffers: Vec<u64> = vec![
            dy_s.buffer,
            b_s.buffer,
            dx.buffer,
            outlier_idx_s.buffer,
            outlier_val_s.buffer,
        ];
        if has_scales {
            buffers.push(scales_s.buffer);
        }

        // --- Lock context and dispatch ---
        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let spirv = spirv_for(kernel).to_vec();
        run_compute_shader(
            ctx,
            &spirv,
            &buffers,
            k.div_ceil(16) as u32,
            m.div_ceil(16) as u32,
            1,
            Some(&push),
        )?;

        Ok((Box::new(dx), Box::new(grim_tensor::backend::ReadyHandle)))
    }
}
