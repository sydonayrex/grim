//! Backward ops implementation for autograd tape entries (WI-T1 item 3).
//!
//! Provides reverse-mode backward implementations for MatMul, Add, Scale, and fused LoRA application.

use grim_tensor::dtype::{BlockDtype, FloatPackScheme, KQuantScheme};
use grim_tensor::{DType, Storage, Tensor, Error, error::Result};
use std::sync::Arc;

/// Arguments for MatMul backward evaluation.
#[derive(Debug, Clone)]
pub struct MatMulArgs {
    pub a: Tensor,
    pub b: Tensor,
    pub out_grad: Tensor,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

/// Arguments for Add backward evaluation.
#[derive(Debug, Clone)]
pub struct AddArgs {
    pub out_grad: Tensor,
}

/// Arguments for Scale backward evaluation.
#[derive(Debug, Clone)]
pub struct ScaleArgs {
    pub input_grad: Tensor,
    pub factor: f32,
}

/// Extract bits-per-weight from a quantized DType storage variant.
fn bpw_from_dtype(dtype: &DType) -> u8 {
    match &dtype.storage {
        Storage::KQuant(scheme) => match scheme {
            KQuantScheme::Q2K => 2,
            KQuantScheme::Q3K => 3,
            KQuantScheme::Q4K => 4,
            KQuantScheme::Q5K => 5,
            KQuantScheme::Q6K => 6,
            KQuantScheme::Q80 => 8,
            KQuantScheme::IQ4NL | KQuantScheme::IQ4XS => 4,
            KQuantScheme::IQ3XXS | KQuantScheme::IQ3S => 3,
            KQuantScheme::IQ2XXS | KQuantScheme::IQ2XS | KQuantScheme::IQ2S => 2,
        },
        Storage::Block(bd) => match bd {
            BlockDtype::Fp4 | BlockDtype::Fp4Block16 => 4,
            BlockDtype::Nf4 => 4,
            BlockDtype::Fp8 | BlockDtype::Fp8Block16 => 8,
        },
        Storage::FloatPack(scheme) => match scheme {
            FloatPackScheme::Fp4 => 4,
            FloatPackScheme::Nf4 => 4,
            FloatPackScheme::Fp8 => 8,
            FloatPackScheme::MxFp4 => 4,
            FloatPackScheme::MxFp8 => 8,
        },
        Storage::GroupInt(cfg) => cfg.bits,
        Storage::Native => 32,
    }
}

/// Compute backward gradients for matrix multiplication `output = A @ B`.
///
/// Returns `(grad_a, grad_b)`. CONTRACT: `out_grad`, `a`, and `b` must have matching dimensions.
pub fn matmul_backward(args: &MatMulArgs) -> Result<(Tensor, Tensor)> {
    let dev = crate::pick_device_for_tensor(&args.out_grad);
    let (a_dims, b_dims) = (args.a.shape().dims(), args.b.shape().dims());

    let (m, k) = match a_dims.len() {
        1 => (1, a_dims[0]),
        _ => if args.transpose_a { (a_dims[1], a_dims[0]) } else { (a_dims[0], a_dims[1]) },
    };
    let (_, n) = match b_dims.len() {
        1 => (b_dims[0], 1),
        _ => if args.transpose_b { (b_dims[1], b_dims[0]) } else { (b_dims[0], b_dims[1]) },
    };

    // Try GPU / ROCm fused backward dispatch when available and b is quantized.
    if !args.transpose_a && !args.transpose_b {
        let b_quantized = matches!(
            args.b.dtype().storage,
            Storage::KQuant(..) | Storage::Block(..) | Storage::FloatPack(..) | Storage::GroupInt(..)
        );
        let b_on_rocm = matches!(args.b.device(), grim_tensor::Device::Rocm(_));

        if b_quantized && b_on_rocm {
            let bpw = bpw_from_dtype(&args.b.dtype());
            let empty_scales: [f32; 0] = [];
            let residuals: Option<grim_tensor::QuantizedMatmulBackwardResiduals> = None;
            if let Ok((grad_a_storage, _handle)) = dev.quantized_matmul_backward_dx(
                args.out_grad.storage().as_ref(),
                args.b.storage().as_ref(),
                &empty_scales,
                bpw,
                m,
                n,
                k,
                args.a.shape(),
                residuals.as_ref(),
            ) {
                let grad_a = Tensor::new(
                    Arc::from(grad_a_storage),
                    args.a.shape().clone(),
                    DType::F32,
                    args.a.provenance().clone(),
                    args.a.device().clone(),
                );

                let (storage_b, _) = dev.matmul(args.a.storage().as_ref(), args.out_grad.storage().as_ref(), args.b.shape())?;
                let grad_b = Tensor::new(
                    Arc::from(storage_b),
                    args.b.shape().clone(),
                    DType::F32,
                    args.b.provenance().clone(),
                    args.b.device().clone(),
                );
                return Ok((grad_a, grad_b));
            }
        }
    }

    // CPU fallback path
    let a_vec = args.a.to_vec_f32()?;
    let b_vec = args.b.to_vec_f32()?;
    let g_vec = args.out_grad.to_vec_f32()?;

    let mut da_vec = vec![0.0f32; a_dims[0] * a_dims[1]];
    let mut db_vec = vec![0.0f32; b_dims[0] * b_dims[1]];

    if !args.transpose_a && !args.transpose_b {
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                for l in 0..n {
                    sum += g_vec[i * n + l] * b_vec[j * n + l];
                }
                da_vec[i * k + j] = sum;
            }
        }
        for i in 0..k {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..m {
                    sum += a_vec[l * k + i] * g_vec[l * n + j];
                }
                db_vec[i * n + j] = sum;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let g = g_vec[i * n + j];
                for l in 0..k {
                    let a_idx = if args.transpose_a { l * m + i } else { i * k + l };
                    let b_idx = if args.transpose_b { j * k + l } else { l * n + j };
                    da_vec[a_idx] += g * b_vec[b_idx];
                    db_vec[b_idx] += g * a_vec[a_idx];
                }
            }
        }
    }

    let storage_a = dev.from_cpu(&da_vec, args.a.shape(), DType::F32)?;
    let grad_a = Tensor::new(
        Arc::from(storage_a),
        args.a.shape().clone(),
        DType::F32,
        args.a.provenance().clone(),
        args.a.device().clone(),
    );

    let storage_b = dev.from_cpu(&db_vec, args.b.shape(), DType::F32)?;
    let grad_b = Tensor::new(
        Arc::from(storage_b),
        args.b.shape().clone(),
        DType::F32,
        args.b.provenance().clone(),
        args.b.device().clone(),
    );

    Ok((grad_a, grad_b))
}

/// Compute backward routing for elementwise add `output = LHS + RHS`.
///
/// Returns `(grad_lhs, grad_rhs)` which are both clones of `out_grad`.
pub fn add_backward(args: &AddArgs) -> Result<(Tensor, Tensor)> {
    Ok((args.out_grad.clone(), args.out_grad.clone()))
}

/// Compute backward gradient for scaling `output = input * factor`.
///
/// Returns `grad_input = out_grad * factor`.
pub fn scale_backward(args: &ScaleArgs) -> Result<Tensor> {
    let dev = crate::pick_device_for_tensor(&args.input_grad);
    let scale_buf = dev.from_cpu(&vec![args.factor; args.input_grad.shape().elem_count()], args.input_grad.shape(), DType::F32)?;
    let (storage, _) = dev.mul(args.input_grad.storage().as_ref(), scale_buf.as_ref(), args.input_grad.shape())?;
    Ok(Tensor::new(
        Arc::from(storage),
        args.input_grad.shape().clone(),
        DType::F32,
        args.input_grad.provenance().clone(),
        args.input_grad.device().clone(),
    ))
}

/// Compute backward gradients for fused LoRA forward pass: `output = base + scale * (x @ A^T) @ B^T`.
///
/// Returns `(grad_base, grad_x, grad_a, grad_b)`.
pub fn lora_backward(
    out_grad: &Tensor,
    x: &Tensor,
    a: &Tensor,
    b: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let grad_base = out_grad.clone();

    let x_vec = x.to_vec_f32()?;
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    let g_vec = out_grad.to_vec_f32()?;

    let x_dims = x.shape().dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    let batch = if x_dims.len() == 1 { 1 } else { x_dims[0] };
    let in_features = if x_dims.len() == 1 { x_dims[0] } else { x_dims[1] };
    let rank = a_dims[0];
    let out_features = b_dims[0];

    let mut h_vec = vec![0.0f32; batch * rank];
    for b_idx in 0..batch {
        for r_idx in 0..rank {
            let mut sum = 0.0f32;
            for i in 0..in_features {
                sum += x_vec[b_idx * in_features + i] * a_vec[r_idx * in_features + i];
            }
            h_vec[b_idx * rank + r_idx] = sum;
        }
    }

    let mut dh_vec = vec![0.0f32; batch * rank];
    for b_idx in 0..batch {
        for r_idx in 0..rank {
            let mut sum = 0.0f32;
            for o in 0..out_features {
                sum += g_vec[b_idx * out_features + o] * b_vec[o * rank + r_idx];
            }
            dh_vec[b_idx * rank + r_idx] = scale * sum;
        }
    }

    let mut db_vec = vec![0.0f32; out_features * rank];
    for o in 0..out_features {
        for r_idx in 0..rank {
            let mut sum = 0.0f32;
            for b_idx in 0..batch {
                sum += g_vec[b_idx * out_features + o] * h_vec[b_idx * rank + r_idx];
            }
            db_vec[o * rank + r_idx] = scale * sum;
        }
    }

    let mut da_vec = vec![0.0f32; rank * in_features];
    for r_idx in 0..rank {
        for i in 0..in_features {
            let mut sum = 0.0f32;
            for b_idx in 0..batch {
                sum += dh_vec[b_idx * rank + r_idx] * x_vec[b_idx * in_features + i];
            }
            da_vec[r_idx * in_features + i] = sum;
        }
    }

    let mut dx_vec = vec![0.0f32; batch * in_features];
    for b_idx in 0..batch {
        for i in 0..in_features {
            let mut sum = 0.0f32;
            for r_idx in 0..rank {
                sum += dh_vec[b_idx * rank + r_idx] * a_vec[r_idx * in_features + i];
            }
            dx_vec[b_idx * in_features + i] = sum;
        }
    }

    let dev = crate::pick_device_for_tensor(out_grad);
    let grad_x = Tensor::new(
        Arc::from(dev.from_cpu(&dx_vec, x.shape(), DType::F32)?),
        x.shape().clone(),
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );
    let grad_a = Tensor::new(
        Arc::from(dev.from_cpu(&da_vec, a.shape(), DType::F32)?),
        a.shape().clone(),
        DType::F32,
        a.provenance().clone(),
        a.device().clone(),
    );
    let grad_b = Tensor::new(
        Arc::from(dev.from_cpu(&db_vec, b.shape(), DType::F32)?),
        b.shape().clone(),
        DType::F32,
        b.provenance().clone(),
        b.device().clone(),
    );

    Ok((grad_base, grad_x, grad_a, grad_b))
}

/// Apply a LoRA adapter to a linear projection output during forward pass and record the operation on `tape`.
///
/// If a LoRA adapter is registered and enabled for `(layer_idx, point)` in `autograd_reg`,
/// this function computes `output = base + scale * (x @ A^T) @ B^T`, registers trainable parameters `A` and `B`
/// on `tape`, and records a `LoRAApply` tape entry. Returns `(output_tensor_id, output_tensor)`.
/// If no adapter is enabled at this point, returns `(base_id, base)`.
pub fn apply_and_record_lora(
    autograd_reg: &crate::registry::AutogradRegistry,
    tape: &mut crate::tape::Tape,
    layer_idx: usize,
    point: crate::injection::LoRAInjectionPoint,
    base: Tensor,
    base_id: crate::tape::TensorId,
    x: Tensor,
    x_id: crate::tape::TensorId,
) -> Result<(crate::tape::TensorId, Tensor)> {
    if let Some(cfg) = autograd_reg.injection_registry.get(layer_idx, point) {
        if cfg.enabled {
            let param_a = autograd_reg
                .params
                .get(cfg.param_id_a())
                .ok_or_else(|| Error::Backend(format!("missing param a for layer {layer_idx} {point:?}")))?;
            let param_b = autograd_reg
                .params
                .get(cfg.param_id_b())
                .ok_or_else(|| Error::Backend(format!("missing param b for layer {layer_idx} {point:?}")))?;

            let a_id = tape.register_param(cfg.param_id_a(), param_a.data.clone());
            let b_id = tape.register_param(cfg.param_id_b(), param_b.data.clone());

            let scale = cfg.scale();
            let dev = crate::pick_device_for_tensor(&base);
            let (out_storage, handle) = dev.lora_accumulate(
                base.storage().as_ref(),
                x.storage().as_ref(),
                param_a.data.storage().as_ref(),
                param_b.data.storage().as_ref(),
                scale,
                base.shape(),
            )?;
            handle.synchronize()?;

            let out_tensor = Tensor::new(
                std::sync::Arc::from(out_storage),
                base.shape().clone(),
                grim_tensor::dtype::DType::F32,
                base.provenance().clone(),
                base.device().clone(),
            );

            let out_id = tape.record_lora_apply(
                base_id,
                x_id,
                a_id,
                b_id,
                out_tensor.clone(),
                cfg.alpha,
                cfg.rank,
                cfg.param_id_a(),
                cfg.param_id_b(),
            );
            return Ok((out_id, out_tensor));
        }
    }
    Ok((base_id, base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        cpu_tensor(data, Shape::new(shape))
    }

    #[test]
    fn scale_backward_multiplies_gradient() {
        let g = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let args = ScaleArgs {
            input_grad: g,
            factor: 0.5,
        };
        let res = scale_backward(&args).unwrap();
        assert_eq!(res.to_vec_f32().unwrap(), vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn add_backward_routes_gradient() {
        let g = tensor(vec![1.0, 2.0], vec![2]);
        let args = AddArgs { out_grad: g };
        let (gl, gr) = add_backward(&args).unwrap();
        assert_eq!(gl.to_vec_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(gr.to_vec_f32().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_gemm_backward() {
        // No transpose: C = A @ B, dA = G @ B^T, dB = A^T @ G
        let a = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]);
        let g = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let args = MatMulArgs {
            a,
            b,
            out_grad: g,
            transpose_a: false,
            transpose_b: false,
        };
        let (ga, gb) = matmul_backward(&args).unwrap();
        assert_eq!(ga.to_vec_f32().unwrap(), vec![2.0, 6.0, 2.0, 6.0]);
        assert_eq!(gb.to_vec_f32().unwrap(), vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_gemm_backward_transpose_a() {
        // Forward: C = A^T @ B, dA_stored = B @ G^T, dB = A @ G
        let a = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]);
        let g = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let args = MatMulArgs {
            a,
            b,
            out_grad: g,
            transpose_a: true,
            transpose_b: false,
        };
        let (ga, gb) = matmul_backward(&args).unwrap();
        assert_eq!(ga.to_vec_f32().unwrap(), vec![2.0, 2.0, 6.0, 6.0]);
        assert_eq!(gb.to_vec_f32().unwrap(), vec![3.0, 3.0, 7.0, 7.0]);
    }

    #[test]
    fn test_gemm_backward_transpose_b() {
        // Forward: C = A @ B^T, dA = G @ B, dB_stored = G^T @ A
        let a = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]);
        let g = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let args = MatMulArgs {
            a,
            b,
            out_grad: g,
            transpose_a: false,
            transpose_b: true,
        };
        let (ga, gb) = matmul_backward(&args).unwrap();
        assert_eq!(ga.to_vec_f32().unwrap(), vec![3.0, 5.0, 3.0, 5.0]);
        assert_eq!(gb.to_vec_f32().unwrap(), vec![4.0, 6.0, 4.0, 6.0]);
    }

    /// Tests that `QuantizedMatmulBackwardResiduals::from_tensor` correctly extracts non-default
    /// residual and outlier metadata from a tensor carrying `QuantProvenance::WithResiduals`.
    #[test]
    fn test_quantized_matmul_backward_residuals_extraction_from_provenance() {
        use grim_tensor::{QuantProvenance, Storage, DType, ArithType, Device};

        let provenance = QuantProvenance::WithResiduals {
            outlier_count: 5,
            outlier_indices_offset: 1024,
            outlier_values_offset: 2048,
            backup1_bpw: 8,
            backup1_codes_offset: 4096,
            backup1_scale_offset: 8192,
            backup2_bpw: 4,
            backup2_codes_offset: 16384,
            backup2_scale_offset: 32768,
        };

        let dummy_storage = grim_backend_cpu::cpu_tensor(vec![0.0f32; 16], Shape::new(vec![4, 4]));
        let b_tensor = Tensor::new(
            dummy_storage.storage().clone(),
            Shape::new(vec![4, 4]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(grim_tensor::KQuantScheme::Q80),
            },
            provenance,
            Device::Cpu,
        );

        let residuals = grim_tensor::QuantizedMatmulBackwardResiduals::from_tensor(&b_tensor);
        assert_eq!(residuals.outlier_count, 5);
        assert_eq!(residuals.backup1_bpw, 8);
        assert_eq!(residuals.backup1_codes_offset, 4096);
        assert_eq!(residuals.backup1_scale_offset, 8192);
        assert_eq!(residuals.backup2_bpw, 4);
        assert_eq!(residuals.backup2_codes_offset, 16384);
        assert_eq!(residuals.backup2_scale_offset, 32768);
    }
}
