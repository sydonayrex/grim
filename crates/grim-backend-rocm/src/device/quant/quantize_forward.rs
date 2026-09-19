//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! Forward-side quantization launchers (q8_0 / fp8 encode, SiLU-mul-quantize).

use std::ffi::c_void;

use grim_tensor::backend::{ ComputeHandle };
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage, Shape };

use crate::device::roc_device::{ RocmDevice };
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, RocmHandle, arg, as_rocm, dev_ptr, dtype_f32, linear_launch };



impl RocmDevice {
    /// Launch standalone Q8_0 quantization HIP kernel.
    pub fn launch_quant_q8_0(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let n_blocks = total.div_ceil(32);
        let grid = crate::HipDim3 {
            x: n_blocks as u32,
            y: 1,
            z: 1,
        };
        let block = crate::HipDim3 { x: 32, y: 1, z: 1 };
        let mut x_ptr = dev_ptr(x)?;
        let mut out_ptr = dev_ptr(out)?;
        let mut total_i = total as i32;

        self.launch_compute_kernel(
            "grim_quant_q8_0",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut total_i)],
        )
    }

    /// Launch standalone FP8 E4M3 quantization HIP kernel.
    pub fn launch_quant_fp8(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let (grid, block) = linear_launch(total);
        let mut x_ptr = dev_ptr(x)?;
        let mut out_ptr = dev_ptr(out)?;
        let mut total_i = total as i32;

        self.launch_compute_kernel(
            "grim_quant_fp8",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut total_i)],
        )
    }

    /// Quantize F32 tensor `x` on-device to `format`.
    pub fn quantize_on_device(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "quantize_on_device: input lacks valid device pointer".into(),
            ));
        }
        let total = x.shape().elem_count();
        use grim_tensor::{FloatPackScheme, KQuantScheme, QuantFormat};
        let (out_bytes, output_dtype) = match format {
            QuantFormat::Q8_0 => {
                let n_blocks = total.div_ceil(32);
                (
                    n_blocks * 34,
                    DType {
                        arith: ArithType::F32,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                )
            }
            QuantFormat::Fp8 => (
                4 + total,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                },
            ),
            other => {
                return Err(Error::Backend(format!(
                    "quantize_on_device: unsupported format {:?}",
                    other
                )));
            }
        };

        let out_shape = x.shape().clone();
        let out_storage = RocmStorage::alloc_gpu_with_bytes(
            &out_shape,
            output_dtype,
            out_bytes,
            &self.allocator,
            self.ordinal,
        )?;

        let stream = match format {
            QuantFormat::Q8_0 => self.launch_quant_q8_0(x_s, &out_storage, total)?,
            QuantFormat::Fp8 => self.launch_quant_fp8(x_s, &out_storage, total)?,
            _ => unreachable!(),
        };

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(stream))),
        ))
    }

    /// Fused 3-in-1 SwiGLU activation + dynamic scale quantization HIP kernel launch.
    pub fn silu_mul_quantize_gpu(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        _format: grim_tensor::dtype::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let g_s = as_rocm(gate)?;
        let u_s = as_rocm(up)?;
        if !g_s.device_ptr_is_valid() || !u_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul_quantize: inputs lack a valid device pointer".into(),
            ));
        }

        let total = out_shape.elem_count();
        let qout_storage = RocmStorage::alloc_gpu(
            out_shape,
            DType {
                arith: grim_tensor::dtype::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let scale_storage = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[1]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut gate_ptr = dev_ptr(g_s)?;
        let mut up_ptr = dev_ptr(u_s)?;
        let mut qout_ptr = dev_ptr(&qout_storage)?;
        let mut scale_ptr = dev_ptr(&scale_storage)?;
        let mut n_i = total as i32;

        let grid = HipDim3::new(1, 1, 1);
        let block = HipDim3::new(256, 1, 1);

        self.launch_compute_kernel(
            "grim_silu_mul_quantize",
            grid,
            block,
            &mut [
                arg(&mut gate_ptr),
                arg(&mut up_ptr),
                arg(&mut qout_ptr),
                arg(&mut scale_ptr),
                arg(&mut n_i),
            ],
        )?;

        Ok((
            Box::new(qout_storage),
            Box::new(scale_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}
