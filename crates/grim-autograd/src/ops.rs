//! Backward ops implementation for autograd tape entries (WI-T1 item 3).
//!
//! Provides reverse-mode backward implementations for MatMul, Add, Scale, and fused LoRA application.

use grim_tensor::{DType, Tensor, error::Result};
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

/// Compute backward gradients for matrix multiplication `output = A @ B`.
///
/// Returns `(grad_a, grad_b)`. CONTRACT: `out_grad`, `a`, and `b` must have matching dimensions.
pub fn matmul_backward(args: &MatMulArgs) -> Result<(Tensor, Tensor)> {
    let dev = crate::pick_device_for_tensor(&args.out_grad);
    let (a_dims, b_dims) = (args.a.shape().dims(), args.b.shape().dims());

    let (m, k) = if args.transpose_a { (a_dims[1], a_dims[0]) } else { (a_dims[0], a_dims[1]) };
    let (_, n) = if args.transpose_b { (b_dims[1], b_dims[0]) } else { (b_dims[0], b_dims[1]) };

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
    let g_vec = args.input_grad.to_vec_f32()?;
    let scaled_vec: Vec<f32> = g_vec.iter().map(|&v| v * args.factor).collect();

    let storage = dev.from_cpu(&scaled_vec, args.input_grad.shape(), DType::F32)?;
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

    let batch = x_dims[0];
    let in_features = x_dims[1];
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
}
