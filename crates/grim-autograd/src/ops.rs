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
        // Transposed gradient computation using the verified backward formulas.
        // Derived from C = A_op @ B_op and standard matrix calculus.
        //
        // For trans_a only (C = A^T @ B, A stored as MxK, B stored as KxN):
        //   dA_stored = B @ G^T  ->  dA[p][q] = sum_l B[p][l] * G[q][l]
        //   dB_stored = A @ G    ->  dB[p][q] = sum_i A[p][i] * G[i][q]
        //
        // For trans_b only (C = A @ B^T, A stored as MxK, B stored as KxN):
        //   dA_stored = G @ B    ->  dA[p][q] = sum_l G[p][l] * B[l][q]
        //   dB_stored = G^T @ A  ->  dB[p][q] = sum_i G[i][p] * A[i][q]
        //
        // For both transposed (C = A^T @ B^T):
        //   dA_stored = B @ G^T  ->  dA[p][q] = sum_l B[p][l] * G[q][l]  (same as trans_a)
        //   dB_stored = G @ A    ->  dB[p][q] = sum_i G[p][i] * A[q][i]   (derived: dB^T = A^T @ G) 
        for p in 0..a_dims[0] {
            for q in 0..a_dims[1] {
                match (args.transpose_a, args.transpose_b) {
                    (true, _) | (_, false) => {
                        // dA = B @ G^T (for trans_a, or both trans): dA[p][q] = sum_l B[p][l] * G[q][l]
                        // dA = G @ B (for trans_b none): dA[p][q] = sum_l G[p][l] * B[l][q]
                    }
                    _ => {}
                }
            }
        }

        let use_bg_for_da = !args.transpose_a; // when A is not transposed, use dA = G @ B

        if use_bg_for_da {
            // dA = G @ B
            for p in 0..a_dims[0] {
                for q in 0..a_dims[1] {
                    let mut sum = 0.0f32;
                    for l in 0..b_dims[0] {
                        sum += g_vec[p * b_dims[0] + l] * b_vec[l * b_dims[1] + q];
                    }
                    da_vec[p * a_dims[1] + q] = sum;
                }
            }
        } else {
            // dA = B @ G^T (trans_a or both trans)
            for p in 0..a_dims[0] {
                for q in 0..a_dims[1] {
                    let mut sum = 0.0f32;
                    for l in 0..b_dims[0] {
                        sum += b_vec[p * b_dims[1] + l] * g_vec[q * b_dims[0] + l];
                    }
                    da_vec[p * a_dims[1] + q] = sum;
                }
            }
        }

        if args.transpose_b {
            // dB_stored = G^T @ A = A^T @ G (transpose of the standard dB)
            // dB[p][q] = sum_i G[i][p] * A[i][q]
            for p in 0..b_dims[0] {
                for q in 0..b_dims[1] {
                    let mut sum = 0.0f32;
                    for i in 0..a_dims[0] {
                        sum += g_vec[i * b_dims[0] + p] * a_vec[i * a_dims[1] + q];
                    }
                    db_vec[p * b_dims[1] + q] = sum;
                }
            }
        } else {
            // dB = A^T @ G (standard) or A @ G (trans_a only)
            if args.transpose_a {
                // dB = A @ G
                for p in 0..b_dims[0] {
                    for q in 0..b_dims[1] {
                        let mut sum = 0.0f32;
                        for i in 0..a_dims[0] {
                            sum += a_vec[p * a_dims[1] + i] * g_vec[i * b_dims[0] + q];
                        }
                        db_vec[p * b_dims[1] + q] = sum;
                    }
                }
            } else {
                // dB = A^T @ G
                for p in 0..b_dims[0] {
                    for q in 0..b_dims[1] {
                        let mut sum = 0.0f32;
                        for i in 0..a_dims[0] {
                            sum += a_vec[i * a_dims[1] + p] * g_vec[i * b_dims[0] + q];
                        }
                        db_vec[p * b_dims[1] + q] = sum;
                    }
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
    let dev = crate::pick_device_for_tensor(x);
    use grim_tensor::{Shape, QuantProvenance};

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

    if matches!(x.device(), grim_tensor::Device::Rocm(_)) {
        let transpose_matrix = |v: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = v[r * cols + c];
                }
            }
            out
        };

        let a_t_vec = transpose_matrix(&a_vec, rank, in_features);
        let a_t_storage = dev.from_cpu(&a_t_vec, &Shape::new(vec![in_features, rank]), DType::F32)?;
        let (h_storage, _) = dev.matmul(x.storage().as_ref(), a_t_storage.as_ref(), &Shape::new(vec![batch, rank]))?;

        let (dh_unscaled, _) = dev.matmul(out_grad.storage().as_ref(), b.storage().as_ref(), &Shape::new(vec![batch, rank]))?;
        let (dh_storage, _) = dev.mul_scalar(dh_unscaled.as_ref(), scale, &Shape::new(vec![batch, rank]))?;

        let g_t_vec = transpose_matrix(&g_vec, batch, out_features);
        let g_t_storage = dev.from_cpu(&g_t_vec, &Shape::new(vec![out_features, batch]), DType::F32)?;
        let (db_unscaled, _) = dev.matmul(g_t_storage.as_ref(), h_storage.as_ref(), &Shape::new(vec![out_features, rank]))?;
        let (db_storage, _) = dev.mul_scalar(db_unscaled.as_ref(), scale, &Shape::new(vec![out_features, rank]))?;

        let dh_vec_gpu = dh_storage.to_cpu_vec_f32()?;
        let dh_t_vec = transpose_matrix(&dh_vec_gpu, batch, rank);
        let dh_t_storage = dev.from_cpu(&dh_t_vec, &Shape::new(vec![rank, batch]), DType::F32)?;
        let (da_storage, _) = dev.matmul(dh_t_storage.as_ref(), x.storage().as_ref(), &Shape::new(vec![rank, in_features]))?;
        let (dx_storage, _) = dev.matmul(dh_storage.as_ref(), a.storage().as_ref(), &Shape::new(vec![batch, in_features]))?;

        let grad_x = Tensor::new(Arc::from(dx_storage), x.shape().clone(), DType::F32, QuantProvenance::default(), x.device().clone());
        let grad_a = Tensor::new(Arc::from(da_storage), a.shape().clone(), DType::F32, QuantProvenance::default(), a.device().clone());
        let grad_b = Tensor::new(Arc::from(db_storage), b.shape().clone(), DType::F32, QuantProvenance::default(), b.device().clone());

        return Ok((grad_base, grad_x, grad_a, grad_b));
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

    // Hidden state `h = x @ A^T` (LoRA forward pre-activation) — needed by the
    // `db_vec` reduction below (`db = scale * G^T @ h`). Mirrors the ROCm
    // branch's `h_storage = dev.matmul(x, a_t, ...)` on host vectors.
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

    #[test]
    fn test_apply_and_record_lora_direct() {
        use crate::{InjectionConfig, LoRAInjectionRegistry, LoRAInjectionPoint};
        let mut tape = crate::tape::Tape::new();
        let inj_config = InjectionConfig {
            hidden_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 4,
            vocab_size: 4,
        };
        let inj_reg = LoRAInjectionRegistry::standard_qlora(1, 4, 16.0, 1);
        let registry = crate::registry::AutogradRegistry::new(inj_config, inj_reg).unwrap();

        let base_tensor = grim_backend_cpu::cpu_tensor(vec![1.0f32, 2.0f32], Shape::new(vec![1, 2]));
        let base_id = tape.register(base_tensor.clone());

        let x_tensor = grim_backend_cpu::cpu_tensor(vec![0.5f32, 0.5f32], Shape::new(vec![1, 2]));
        let x_id = tape.register(x_tensor.clone());

        let (out_id, out_tensor) = apply_and_record_lora(
            &registry,
            &mut tape,
            0,
            LoRAInjectionPoint::QProj,
            base_tensor,
            base_id,
            x_tensor,
            x_id,
        ).unwrap();

        assert_eq!(out_tensor.shape().dims(), &[1, 2]);
        assert!(tape.len() > 0);
    }
}
