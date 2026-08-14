//! Backward ops implementation for autograd tape entries (WI-T1 item 3).
//!
//! Provides reverse-mode backward implementations for MatMul, Add, Scale, and fused LoRA application.

use grim_tensor::dtype::{BlockDtype, FloatPackScheme, KQuantScheme};
use grim_tensor::{DType, Device, Error, Shape, Storage, Tensor, error::Result};
use std::sync::Arc;

/// DoRA (Weight-Decomposed LoRA) forward pass.
///
/// Implements the weight-decomposed LoRA from the Unsloth/DoRA paper.
///
/// Mathematical formulas from the plan spec:
/// 1. Directional matrix: V = W_0 + γ * B @ A
/// 2. Column-wise L2 norm for output row i: n_i = ||V[i,:]||_2 = sqrt(sum_j V[i,j]^2 + ε)
/// 3. Normalized directional matrix: V_hat[i,j] = V[i,j] / n_i
/// 4. Effective weight: W_eff[i,j] = m_i * V_hat[i,j]
/// 5. Output for input X: Y = X @ W_eff^T
///
/// Returns Y = X @ W_eff^T.

pub fn dora_forward(
    x: &Tensor,
    w_base: &Tensor,
    a: &Tensor,
    b: &Tensor,
    m: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let dev = crate::pick_device_for_tensor(x);

    // Get shapes
    let x_dims = x.shape().dims();
    let w_dims = w_base.shape().dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let m_dims = m.shape().dims();

    // Handle batch dimension
    let batch = if x_dims.len() == 1 { 1 } else { x_dims[0] };
    let in_features = if x_dims.len() == 1 {
        x_dims[0]
    } else {
        x_dims[1]
    };

    // Validate dimensions
    let out_features = w_dims[0];
    let rank = a_dims[0];
    let b_out_features = b_dims[0];

    if out_features != b_out_features {
        return Err(Error::Backend(format!(
            "DoRA: w_base and b output feature mismatch: {} vs {}",
            out_features, b_out_features
        )));
    }
    if m_dims[0] != out_features {
        return Err(Error::Backend(format!(
            "DoRA: m size {} != out_features {}",
            m_dims[0], out_features
        )));
    }
    if a_dims[1] != in_features {
        return Err(Error::Backend(format!(
            "DoRA: a input feature {} != x input feature {}",
            a_dims[1], in_features
        )));
    }

    // Get data as vectors for CPU computation
    let x_vec = x.to_vec_f32()?;
    let w_vec = w_base.to_vec_f32()?;
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    let m_vec = m.to_vec_f32()?;

    // Step 1: Compute BA = B @ A (shape: [out_features, in_features])
    let mut ba_vec = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += b_vec[i * rank + r] * a_vec[r * in_features + j];
            }
            ba_vec[i * in_features + j] = sum;
        }
    }

    // Step 2: Compute V = W_0 + scale * BA
    let v_vec: Vec<f32> = w_vec
        .iter()
        .zip(ba_vec.iter())
        .map(|(&w, &ba)| w + scale * ba)
        .collect();

    // Step 3: Compute norm vector n_i = sqrt(sum_j V[i,j]^2 + eps)
    let eps = 1e-8f32;
    let mut n_vec = vec![0.0f32; out_features];
    for i in 0..out_features {
        let mut sum_sq = 0.0f32;
        for j in 0..in_features {
            sum_sq += v_vec[i * in_features + j] * v_vec[i * in_features + j];
        }
        n_vec[i] = (sum_sq + eps).sqrt();
    }

    // Step 4: Compute V_hat[i,j] = V[i,j] / n_i
    let v_hat_vec: Vec<f32> = (0..out_features * in_features)
        .map(|idx| {
            let i = idx / in_features;
            let j = idx % in_features;
            v_vec[i * in_features + j] / n_vec[i]
        })
        .collect();

    // Step 5: Compute W_eff[i,j] = m[i] * V_hat[i,j]
    let w_eff_vec: Vec<f32> = v_hat_vec
        .iter()
        .enumerate()
        .map(|(idx, &v_hat)| {
            let i = idx / in_features;
            m_vec[i] * v_hat
        })
        .collect();

    // Step 6: Compute Y = X @ W_eff^T = (X @ W_eff^T)
    // Output shape: [batch, out_features]
    let mut y_vec = vec![0.0f32; batch * out_features];
    for b_idx in 0..batch {
        for o in 0..out_features {
            let mut sum = 0.0f32;
            for i in 0..in_features {
                sum += x_vec[b_idx * in_features + i] * w_eff_vec[o * in_features + i];
            }
            y_vec[b_idx * out_features + o] = sum;
        }
    }

    let out_shape = if batch == 1 {
        Shape::new(vec![out_features])
    } else {
        Shape::new(vec![batch, out_features])
    };

    let storage = dev.from_cpu(&y_vec, &out_shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        out_shape,
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    ))
}

/// DoRA backward pass.
///
/// Computes gradients w.r.t. all inputs:
/// - ∇W_eff = out_grad^T @ X
/// - ∇m_i = sum_j (∇W_eff[i,j] * V_hat[i,j])
/// - ∇V[i,j] = (m_i / n_i) * (∇W_eff[i,j] - V_hat[i,j] * sum_k(∇W_eff[i,k] * V_hat[i,k]))
/// - ∇B = scale * (∇V @ A^T)
/// - ∇A = scale * (B^T @ ∇V)
/// - ∇X = out_grad @ W_eff
pub fn dora_backward(
    out_grad: &Tensor,
    x: &Tensor,
    w_base: &Tensor,
    a: &Tensor,
    b: &Tensor,
    m: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let dev = crate::pick_device_for_tensor(x);

    // Get shapes
    let x_dims = x.shape().dims();
    let w_dims = w_base.shape().dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let m_dims = m.shape().dims();

    // Handle batch dimension
    let batch = if x_dims.len() == 1 { 1 } else { x_dims[0] };
    let in_features = if x_dims.len() == 1 {
        x_dims[0]
    } else {
        x_dims[1]
    };

    let out_features = w_dims[0];
    let rank = a_dims[0];

    // Get data as vectors
    let x_vec = x.to_vec_f32()?;
    let w_vec = w_base.to_vec_f32()?;
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    let m_vec = m.to_vec_f32()?;
    let g_vec = out_grad.to_vec_f32()?;

    // Recompute V and V_hat (needed for backward) - same as forward
    let eps = 1e-8f32;

    // Compute BA = B @ A
    let mut ba_vec = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += b_vec[i * rank + r] * a_vec[r * in_features + j];
            }
            ba_vec[i * in_features + j] = sum;
        }
    }

    // V = W_0 + scale * BA
    let v_vec: Vec<f32> = w_vec
        .iter()
        .zip(ba_vec.iter())
        .map(|(&w, &ba)| w + scale * ba)
        .collect();

    // Compute norm vector n_i
    let mut n_vec = vec![0.0f32; out_features];
    for i in 0..out_features {
        let mut sum_sq = 0.0f32;
        for j in 0..in_features {
            sum_sq += v_vec[i * in_features + j] * v_vec[i * in_features + j];
        }
        n_vec[i] = (sum_sq + eps).sqrt();
    }

    // V_hat[i,j] = V[i,j] / n_i
    let v_hat_vec: Vec<f32> = (0..out_features * in_features)
        .map(|idx| {
            let i = idx / in_features;
            let j = idx % in_features;
            v_vec[i * in_features + j] / n_vec[i]
        })
        .collect();

    // Step 1: ∇W_eff = out_grad^T @ X
    // out_grad shape: [batch, out_features], X shape: [batch, in_features]
    // grad_w_eff shape: [out_features, in_features]
    let mut grad_w_eff_vec = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for b_idx in 0..batch {
                sum += g_vec[b_idx * out_features + i] * x_vec[b_idx * in_features + j];
            }
            grad_w_eff_vec[i * in_features + j] = sum;
        }
    }

    // Step 2: ∇m_i = sum_j (∇W_eff[i,j] * V_hat[i,j])
    let mut grad_m_vec = vec![0.0f32; out_features];
    for i in 0..out_features {
        let mut sum = 0.0f32;
        for j in 0..in_features {
            sum += grad_w_eff_vec[i * in_features + j] * v_hat_vec[i * in_features + j];
        }
        grad_m_vec[i] = sum;
    }

    // Step 3: ∇V[i,j] = (m_i / n_i) * (∇W_eff[i,j] - V_hat[i,j] * sum_k(∇W_eff[i,k] * V_hat[i,k]))
    let mut grad_v_vec = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        // Compute sum_k(∇W_eff[i,k] * V_hat[i,k])
        let mut sum_k = 0.0f32;
        for k in 0..in_features {
            sum_k += grad_w_eff_vec[i * in_features + k] * v_hat_vec[i * in_features + k];
        }
        for j in 0..in_features {
            grad_v_vec[i * in_features + j] = (m_vec[i] / n_vec[i])
                * (grad_w_eff_vec[i * in_features + j] - v_hat_vec[i * in_features + j] * sum_k);
        }
    }

    // Step 4: ∇B = scale * (∇V @ A^T)
    // grad_v shape: [out_features, in_features], A shape: [rank, in_features]
    // grad_b shape: [out_features, rank]
    let mut grad_b_vec = vec![0.0f32; out_features * rank];
    for i in 0..out_features {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for j in 0..in_features {
                sum += grad_v_vec[i * in_features + j] * a_vec[r * in_features + j];
            }
            grad_b_vec[i * rank + r] = scale * sum;
        }
    }

    // Step 5: ∇A = scale * (B^T @ ∇V)
    // grad_v shape: [out_features, in_features], B shape: [out_features, rank]
    // grad_a shape: [rank, in_features]
    let mut grad_a_vec = vec![0.0f32; rank * in_features];
    for r in 0..rank {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for i in 0..out_features {
                sum += b_vec[i * rank + r] * grad_v_vec[i * in_features + j];
            }
            grad_a_vec[r * in_features + j] = scale * sum;
        }
    }

    // Step 6: ∇X = out_grad @ W_eff
    // out_grad shape: [batch, out_features], W_eff shape: [out_features, in_features]
    // grad_x shape: [batch, in_features]
    let mut grad_x_vec = vec![0.0f32; batch * in_features];
    for b_idx in 0..batch {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for i in 0..out_features {
                sum += g_vec[b_idx * out_features + i] * grad_w_eff_vec[i * in_features + j];
            }
            grad_x_vec[b_idx * in_features + j] = sum;
        }
    }

    // Create output tensors
    let out_shape_x = if batch == 1 {
        Shape::new(vec![in_features])
    } else {
        Shape::new(vec![batch, in_features])
    };
    let out_shape_w = Shape::new(w_dims);
    let out_shape_a = Shape::new(a_dims);
    let out_shape_b = Shape::new(b_dims);
    let out_shape_m = Shape::new(m_dims);

    let grad_x = Tensor::new(
        Arc::from(dev.from_cpu(&grad_x_vec, &out_shape_x, DType::F32)?),
        out_shape_x,
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );

    // For DoRA, the base weight w_base is frozen (not trainable), so grad_w_base is zeros
    // matching the original stub signature: (grad_x, grad_w_base, grad_a, grad_b, grad_m)
    let z_w_vec = vec![0.0f32; w_vec.len()];
    let grad_w_base = Tensor::new(
        Arc::from(dev.from_cpu(&z_w_vec, &out_shape_w, DType::F32)?),
        out_shape_w.clone(),
        DType::F32,
        w_base.provenance().clone(),
        w_base.device().clone(),
    );

    let grad_a = Tensor::new(
        Arc::from(dev.from_cpu(&grad_a_vec, &out_shape_a, DType::F32)?),
        out_shape_a,
        DType::F32,
        a.provenance().clone(),
        a.device().clone(),
    );
    let grad_b = Tensor::new(
        Arc::from(dev.from_cpu(&grad_b_vec, &out_shape_b, DType::F32)?),
        out_shape_b,
        DType::F32,
        b.provenance().clone(),
        b.device().clone(),
    );
    let grad_m = Tensor::new(
        Arc::from(dev.from_cpu(&grad_m_vec, &out_shape_m, DType::F32)?),
        out_shape_m,
        DType::F32,
        m.provenance().clone(),
        m.device().clone(),
    );

    Ok((grad_x, grad_w_base, grad_a, grad_b, grad_m))
}

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
        Storage::ResidualPacked(cfg) => cfg.bpw,
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
        _ => {
            if args.transpose_a {
                (a_dims[1], a_dims[0])
            } else {
                (a_dims[0], a_dims[1])
            }
        }
    };
    let (_, n) = match b_dims.len() {
        1 => (b_dims[0], 1),
        _ => {
            if args.transpose_b {
                (b_dims[1], b_dims[0])
            } else {
                (b_dims[0], b_dims[1])
            }
        }
    };

    // Try GPU fused backward dispatch when available and b is quantized.
    if !args.transpose_a && !args.transpose_b {
        let b_quantized = matches!(
            args.b.dtype().storage,
            Storage::KQuant(..)
                | Storage::Block(..)
                | Storage::FloatPack(..)
                | Storage::GroupInt(..)
        );
        let b_on_rocm = matches!(args.b.device(), grim_tensor::Device::Rocm(_));
        let b_on_cuda = matches!(args.b.device(), grim_tensor::Device::Cuda(_));
        let b_on_vulkan = matches!(args.b.device(), grim_tensor::Device::Vulkan);
        let b_on_metal = matches!(args.b.device(), grim_tensor::Device::Metal(_));

        let empty_scales: [f32; 0] = [];
        let b_scales: &[f32] = match args.b.dtype().storage {
            Storage::KQuant(..) | Storage::Block(..) | Storage::FloatPack(..) => &empty_scales,
            Storage::ResidualPacked(..) | Storage::GroupInt(..) => match args.b.quant_scales() {
                Some(s) => s,
                None => {
                    return Err(Error::Backend(
                        "matmul_backward: ResidualPacked/GroupInt tensor requires explicit scales"
                            .into(),
                    ));
                }
            },
            Storage::Native => &empty_scales,
        };

        if b_quantized && (b_on_rocm || b_on_cuda || b_on_vulkan || b_on_metal) {
            let bpw = bpw_from_dtype(&args.b.dtype());
            let residuals = grim_tensor::QuantizedMatmulBackwardResiduals::from_tensor(&args.b);
            if let Ok((grad_a_storage, _handle)) = dev.quantized_matmul_backward_dx(
                args.out_grad.storage().as_ref(),
                args.b.storage().as_ref(),
                b_scales,
                bpw,
                m,
                n,
                k,
                args.a.shape(),
                Some(&residuals),
            ) {
                let grad_a = Tensor::new(
                    Arc::from(grad_a_storage),
                    args.a.shape().clone(),
                    DType::F32,
                    args.a.provenance().clone(),
                    args.a.device().clone(),
                );

                let (storage_b, _) = dev.matmul(
                    args.a.storage().as_ref(),
                    args.out_grad.storage().as_ref(),
                    args.b.shape(),
                )?;
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
        // dA = G @ B^T  where G is [m,n], B is [k,n], dA is [m,k].
        // dA[i,j] = sum_l G[i,l] * B[j,l]
        // Loop order i,j,l keeps both g_vec and b_vec sequential in l (cache-friendly).
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                let g_base = i * n;
                let b_base = j * n;
                for l in 0..n {
                    sum += g_vec[g_base + l] * b_vec[b_base + l];
                }
                da_vec[i * k + j] = sum;
            }
        }
        // dB = A^T @ G  where A is [m,k], G is [m,n], dB is [k,n].
        // dB[i,j] = sum_l A[l,i] * G[l,j]
        // Loop order i,j,l keeps both a_vec and g_vec sequential in l.
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
        for _p in 0..a_dims[0] {
            for _q in 0..a_dims[1] {
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
            // dA = G @ B,  dA[p][q] = sum_l G[p][l] * B[l][q]
            // Reordered to p,l,q so G[p][l] is hoisted and B[l][q] is sequential in q.
            let m = a_dims[0];
            let k = a_dims[1];
            let bn = b_dims[1];
            for p in 0..m {
                for q in 0..k {
                    da_vec[p * k + q] = 0.0;
                }
                for l in 0..b_dims[0] {
                    let gl = g_vec[p * b_dims[0] + l];
                    let b_row = &b_vec[l * bn..];
                    for q in 0..k {
                        da_vec[p * k + q] += gl * b_row[q];
                    }
                }
            }
        } else {
            // dA = B @ G^T,  dA[p][q] = sum_l B[p][l] * G[q][l]
            // Reordered to p,q,l so B[p][l] and G[q][l] are both sequential in l.
            for p in 0..a_dims[0] {
                for q in 0..a_dims[1] {
                    let mut sum = 0.0f32;
                    let g_base = q * b_dims[0];
                    for l in 0..b_dims[0] {
                        sum += b_vec[p * b_dims[1] + l] * g_vec[g_base + l];
                    }
                    da_vec[p * a_dims[1] + q] = sum;
                }
            }
        }

        if args.transpose_b {
            // dB_stored = G^T @ A = A^T @ G  (transpose of the standard dB)
            // dB[p][q] = sum_i G[i][p] * A[i][q]
            // Reordered to i,p,q so A[i][q] and G[i][p] are sequential in p for fixed i.
            for i in 0..a_dims[0] {
                let g_base = i * b_dims[0];
                let a_base = i * a_dims[1];
                for p in 0..b_dims[0] {
                    let gp = g_vec[g_base + p];
                    for q in 0..b_dims[1] {
                        db_vec[p * b_dims[1] + q] += gp * a_vec[a_base + q];
                    }
                }
            }
        } else {
            // dB = A^T @ G (standard) or A @ G (trans_a only)
            if args.transpose_a {
                // dB = A @ G  =>  dB[p][q] = sum_i A[p][i] * G[i][q]
                // Reordered to p,i,q so G[i][q] is sequential in q, A[p][i] hoisted.
                for p in 0..b_dims[0] {
                    for q in 0..b_dims[1] {
                        db_vec[p * b_dims[1] + q] = 0.0;
                    }
                    for i in 0..a_dims[0] {
                        let a_val = a_vec[p * a_dims[1] + i];
                        let g_row = &g_vec[i * b_dims[0]..];
                        for q in 0..b_dims[1] {
                            db_vec[p * b_dims[1] + q] += a_val * g_row[q];
                        }
                    }
                }
            } else {
                // dB = A^T @ G  =>  dB[p][q] = sum_i A[i][p] * G[i][q]
                // Reordered to i,p,q: A[i][p] sequential in p, G[i][q] sequential in q.
                for i in 0..a_dims[0] {
                    let a_base = i * a_dims[1];
                    let g_base = i * b_dims[0];
                    for p in 0..b_dims[0] {
                        let ap = a_vec[a_base + p];
                        for q in 0..b_dims[1] {
                            db_vec[p * b_dims[1] + q] += ap * g_vec[g_base + q];
                        }
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
    let scale_buf = dev.from_cpu(
        &vec![args.factor; args.input_grad.shape().elem_count()],
        args.input_grad.shape(),
        DType::F32,
    )?;
    let (storage, _) = dev.mul(
        args.input_grad.storage().as_ref(),
        scale_buf.as_ref(),
        args.input_grad.shape(),
    )?;
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
    use grim_tensor::{QuantProvenance, Shape};

    let x_vec = x.to_vec_f32()?;
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    let g_vec = out_grad.to_vec_f32()?;

    let x_dims = x.shape().dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    let batch = if x_dims.len() == 1 { 1 } else { x_dims[0] };
    let in_features = if x_dims.len() == 1 {
        x_dims[0]
    } else {
        x_dims[1]
    };
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
        let a_t_storage =
            dev.from_cpu(&a_t_vec, &Shape::new(vec![in_features, rank]), DType::F32)?;
        let (h_storage, _) = dev.matmul(
            x.storage().as_ref(),
            a_t_storage.as_ref(),
            &Shape::new(vec![batch, rank]),
        )?;

        let (dh_unscaled, _) = dev.matmul(
            out_grad.storage().as_ref(),
            b.storage().as_ref(),
            &Shape::new(vec![batch, rank]),
        )?;
        let (dh_storage, _) =
            dev.mul_scalar(dh_unscaled.as_ref(), scale, &Shape::new(vec![batch, rank]))?;

        let g_t_vec = transpose_matrix(&g_vec, batch, out_features);
        let g_t_storage =
            dev.from_cpu(&g_t_vec, &Shape::new(vec![out_features, batch]), DType::F32)?;
        let (db_unscaled, _) = dev.matmul(
            g_t_storage.as_ref(),
            h_storage.as_ref(),
            &Shape::new(vec![out_features, rank]),
        )?;
        let (db_storage, _) = dev.mul_scalar(
            db_unscaled.as_ref(),
            scale,
            &Shape::new(vec![out_features, rank]),
        )?;

        let dh_vec_gpu = dh_storage.to_cpu_vec_f32()?;
        let dh_t_vec = transpose_matrix(&dh_vec_gpu, batch, rank);
        let dh_t_storage = dev.from_cpu(&dh_t_vec, &Shape::new(vec![rank, batch]), DType::F32)?;
        let (da_storage, _) = dev.matmul(
            dh_t_storage.as_ref(),
            x.storage().as_ref(),
            &Shape::new(vec![rank, in_features]),
        )?;
        let (dx_storage, _) = dev.matmul(
            dh_storage.as_ref(),
            a.storage().as_ref(),
            &Shape::new(vec![batch, in_features]),
        )?;

        let grad_x = Tensor::new(
            Arc::from(dx_storage),
            x.shape().clone(),
            DType::F32,
            QuantProvenance::default(),
            x.device().clone(),
        );
        let grad_a = Tensor::new(
            Arc::from(da_storage),
            a.shape().clone(),
            DType::F32,
            QuantProvenance::default(),
            a.device().clone(),
        );
        let grad_b = Tensor::new(
            Arc::from(db_storage),
            b.shape().clone(),
            DType::F32,
            QuantProvenance::default(),
            b.device().clone(),
        );

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

/// SwiGLU backward: `output = silu(gate) * up`.
///
/// Returns `(d_gate, d_up)` where:
/// - `d_gate = dw * up * silu'(gate)`
/// - `d_up = dw * silu(gate)`
///
/// On ROCm, dispatches to the `grim_silu_mul_backward` HIP kernel for
/// device-resident computation. Falls back to CPU vector math otherwise.
pub fn silu_mul_backward(gate: &Tensor, up: &Tensor, dw: &Tensor) -> Result<(Tensor, Tensor)> {
    let n = gate.shape().elem_count();
    let dev = crate::pick_device_for_tensor(gate);

    // ROCm GPU path: use the fused silu_mul_backward kernel.
    if let Device::Rocm(_) = gate.device() {
        let gate_s = gate.storage().as_ref();
        let up_s = up.storage().as_ref();
        let dw_s = dw.storage().as_ref();
        if let Ok((df_storage, de_storage, _handle)) =
            dev.silu_mul_backward(gate_s, up_s, dw_s, gate.shape())
        {
            let d_gate = Tensor::new(
                Arc::from(df_storage),
                gate.shape().clone(),
                DType::F32,
                gate.provenance().clone(),
                gate.device().clone(),
            );
            let d_up = Tensor::new(
                Arc::from(de_storage),
                up.shape().clone(),
                DType::F32,
                up.provenance().clone(),
                up.device().clone(),
            );
            return Ok((d_gate, d_up));
        }
    }

    // CPU fallback: elementwise SwiGLU backward.
    let gate_vec = gate.to_vec_f32()?;
    let up_vec = up.to_vec_f32()?;
    let dw_vec = dw.to_vec_f32()?;

    let mut d_gate_vec = vec![0.0f32; n];
    let mut d_up_vec = vec![0.0f32; n];
    for i in 0..n {
        let g = gate_vec[i];
        let se = 1.0f32 / (1.0f32 + (-g).exp());
        let silu_g = se * g;
        let dsilu = se * (1.0f32 + g * (1.0f32 - se));
        d_gate_vec[i] = dw_vec[i] * up_vec[i] * dsilu;
        d_up_vec[i] = dw_vec[i] * silu_g;
    }

    let d_gate = Tensor::new(
        Arc::from(dev.from_cpu(&d_gate_vec, gate.shape(), DType::F32)?),
        gate.shape().clone(),
        DType::F32,
        gate.provenance().clone(),
        gate.device().clone(),
    );
    let d_up = Tensor::new(
        Arc::from(dev.from_cpu(&d_up_vec, up.shape(), DType::F32)?),
        up.shape().clone(),
        DType::F32,
        up.provenance().clone(),
        up.device().clone(),
    );

    Ok((d_gate, d_up))
}
///
/// Computes a codebook-quantized low-rank update. `BA = B @ A` is quantized per
/// output row against a set of scalar codebooks before being applied, so the
/// effective adaptation only ever reads from a small centroid table.
///
/// Steps:
/// 1. `BA = B @ A`, shape `[d_out, d_in]`.
/// 2. Each output row `i` is assigned a codebook `q(i) = i % num_codebooks`.
/// 3. `index_ij = argmin_k |BA[i,j] - codebook[q(i)][k]|` (nearest centroid).
/// 4. `ĥBA[i,j] = codebook[q(i)][index_ij]`.
/// 5. `Y = scale * X @ ĥBA^T` (LoRA-style delta; the caller adds the base output).
pub fn vera_forward(
    x: &Tensor,
    a: &Tensor,
    b: &Tensor,
    codebook: &[Vec<f32>],
    scale: f32,
    num_codebooks: usize,
) -> Result<Tensor> {
    let dev = crate::pick_device_for_tensor(x);

    let x_dims = x.shape().dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    let batch = if x_dims.len() == 1 { 1 } else { x_dims[0] };
    let in_features = if x_dims.len() == 1 {
        x_dims[0]
    } else {
        x_dims[1]
    };
    let rank = a_dims[0];
    let out_features = b_dims[0];

    if codebook.is_empty() || num_codebooks == 0 {
        return Err(Error::Backend("VeRA: empty codebook set".into()));
    }
    if a_dims[1] != in_features || b_dims[1] != rank {
        return Err(Error::Backend("VeRA: A/B shape mismatch".into()));
    }

    let x_vec = x.to_vec_f32()?;
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;

    // Step 1: BA = B @ A.
    let mut ba = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += b_vec[i * rank + r] * a_vec[r * in_features + j];
            }
            ba[i * in_features + j] = sum;
        }
    }

    // Steps 2-4: nearest-centroid quantization per output row.
    let ncb = num_codebooks.min(codebook.len());
    let mut ba_hat = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        let cb = &codebook[i % ncb];
        for j in 0..in_features {
            let v = ba[i * in_features + j];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (k, &c) in cb.iter().enumerate() {
                let d = (v - c).abs();
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            ba_hat[i * in_features + j] = cb[best];
        }
    }

    // Step 5: Y = scale * X @ ĥBA^T.
    let mut y_vec = vec![0.0f32; batch * out_features];
    for b_idx in 0..batch {
        for o in 0..out_features {
            let mut sum = 0.0f32;
            for j in 0..in_features {
                sum += x_vec[b_idx * in_features + j] * ba_hat[o * in_features + j];
            }
            y_vec[b_idx * out_features + o] = scale * sum;
        }
    }

    let out_shape = if batch == 1 {
        Shape::new(vec![out_features])
    } else {
        Shape::new(vec![batch, out_features])
    };
    let storage = dev.from_cpu(&y_vec, &out_shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        out_shape,
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    ))
}

/// VeRA backward pass (straight-through estimator).
///
/// The quantize/dequantize step is treated as the identity for gradient flow:
/// the gradient of the dequantized update `ĥBA` is passed straight through to
/// `BA`, and from there to `A` and `B`. Codebook centroids receive the gradient
/// of the elements assigned to them (the "quantize gradient before updating
/// codebook vectors" rule).
///
/// Returns `(grad_base, grad_x, grad_a, grad_b, grad_codebook)` where
/// `grad_base = out_grad` (base-weight path) and `grad_codebook` has shape
/// `[num_codebooks, codebook_size]` (flattened centroid gradients per codebook;
/// all used codebooks must have equal length).
pub fn vera_backward(
    out_grad: &Tensor,
    x: &Tensor,
    a: &Tensor,
    b: &Tensor,
    codebook: &[Vec<f32>],
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let dev = crate::pick_device_for_tensor(x);

    let x_dims = x.shape().dims();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let g_dims = out_grad.shape().dims();

    let batch = if x_dims.len() == 1 { 1 } else { x_dims[0] };
    let in_features = if x_dims.len() == 1 {
        x_dims[0]
    } else {
        x_dims[1]
    };
    let rank = a_dims[0];
    let out_features = b_dims[0];
    let g_out = if g_dims.len() == 1 { 1 } else { g_dims[0] };

    if codebook.is_empty() {
        return Err(Error::Backend("VeRA backward: empty codebook set".into()));
    }
    if g_out != batch || g_dims[g_dims.len() - 1] != out_features {
        return Err(Error::Backend(
            "VeRA backward: out_grad shape mismatch".into(),
        ));
    }
    if a_dims[1] != in_features || b_dims[1] != rank {
        return Err(Error::Backend("VeRA backward: A/B shape mismatch".into()));
    }

    let ncb = codebook.len();
    let codebook_size = codebook[0].len();
    if codebook.iter().any(|cb| cb.len() != codebook_size) {
        return Err(Error::Backend("VeRA backward: ragged codebooks".into()));
    }

    let x_vec = x.to_vec_f32()?;
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    let g_vec = out_grad.to_vec_f32()?;

    // Recompute BA, ĥBA and centroid assignments (deterministic recompute).
    let mut ba = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += b_vec[i * rank + r] * a_vec[r * in_features + j];
            }
            ba[i * in_features + j] = sum;
        }
    }
    let mut ba_hat = vec![0.0f32; out_features * in_features];
    let mut assign = vec![0usize; out_features * in_features];
    for i in 0..out_features {
        let cb = &codebook[i % ncb];
        for j in 0..in_features {
            let v = ba[i * in_features + j];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (k, &c) in cb.iter().enumerate() {
                let d = (v - c).abs();
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            assign[i * in_features + j] = best;
            ba_hat[i * in_features + j] = cb[best];
        }
    }

    // ∇ĥBA[o,j] = scale * Σ_b X[b,j] * out_grad[b,o]. STE: ∇BA = ∇ĥBA.
    let mut grad_ba = vec![0.0f32; out_features * in_features];
    for o in 0..out_features {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for b_idx in 0..batch {
                sum += x_vec[b_idx * in_features + j] * g_vec[b_idx * out_features + o];
            }
            grad_ba[o * in_features + j] = scale * sum;
        }
    }

    // ∇B[o,r] = Σ_j ∇BA[o,j] * A[r,j]
    let mut grad_b = vec![0.0f32; out_features * rank];
    for o in 0..out_features {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for j in 0..in_features {
                sum += grad_ba[o * in_features + j] * a_vec[r * in_features + j];
            }
            grad_b[o * rank + r] = sum;
        }
    }

    // ∇A[r,j] = Σ_o B[o,r] * ∇BA[o,j]
    let mut grad_a = vec![0.0f32; rank * in_features];
    for r in 0..rank {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for o in 0..out_features {
                sum += b_vec[o * rank + r] * grad_ba[o * in_features + j];
            }
            grad_a[r * in_features + j] = sum;
        }
    }

    // ∇X[b,j] = scale * Σ_o out_grad[b,o] * ĥBA[o,j]
    let mut grad_x = vec![0.0f32; batch * in_features];
    for b_idx in 0..batch {
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for o in 0..out_features {
                sum += g_vec[b_idx * out_features + o] * ba_hat[o * in_features + j];
            }
            grad_x[b_idx * in_features + j] = scale * sum;
        }
    }

    // ∇codebook[g][k] = Σ over elements (i,j) assigned to centroid k of codebook g.
    let mut grad_cb = vec![0.0f32; ncb * codebook_size];
    for i in 0..out_features {
        let g = i % ncb;
        for j in 0..in_features {
            let k = assign[i * in_features + j];
            grad_cb[g * codebook_size + k] += grad_ba[i * in_features + j];
        }
    }

    let mk = |v: Vec<f32>, shape: &Shape, orig: &Tensor| -> Result<Tensor> {
        Ok(Tensor::new(
            Arc::from(dev.from_cpu(&v, shape, DType::F32)?),
            shape.clone(),
            DType::F32,
            orig.provenance().clone(),
            orig.device().clone(),
        ))
    };

    let grad_base = out_grad.clone();
    let grad_x = mk(grad_x, x.shape(), x)?;
    let grad_a = mk(grad_a, a.shape(), a)?;
    let grad_b = mk(grad_b, b.shape(), b)?;
    let grad_codebook = mk(grad_cb, &Shape::new(vec![ncb, codebook_size]), out_grad)?;

    Ok((grad_base, grad_x, grad_a, grad_b, grad_codebook))
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
            let param_a = autograd_reg.params.get(cfg.param_id_a()).ok_or_else(|| {
                Error::Backend(format!("missing param a for layer {layer_idx} {point:?}"))
            })?;
            let param_b = autograd_reg.params.get(cfg.param_id_b()).ok_or_else(|| {
                Error::Backend(format!("missing param b for layer {layer_idx} {point:?}"))
            })?;

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

/// Fused SwiGLU + LoRA MLP activation forward pass helper.
///
/// Evaluates Gate and Up linear projections with LoRA adapters (if registered),
/// applies SwiGLU activation (`silu(gate) * up`), and registers the resulting activation on `tape`.
pub fn fused_swiglu_lora_mlp(
    autograd_reg: &crate::registry::AutogradRegistry,
    tape: &mut crate::tape::Tape,
    layer_idx: usize,
    gate_base: Tensor,
    gate_base_id: crate::tape::TensorId,
    up_base: Tensor,
    up_base_id: crate::tape::TensorId,
    x: Tensor,
    x_id: crate::tape::TensorId,
) -> Result<(crate::tape::TensorId, Tensor)> {
    let (gate_id, gate_out) = apply_and_record_lora(
        autograd_reg,
        tape,
        layer_idx,
        crate::injection::LoRAInjectionPoint::GateProj,
        gate_base,
        gate_base_id,
        x.clone(),
        x_id,
    )?;

    let (up_id, up_out) = apply_and_record_lora(
        autograd_reg,
        tape,
        layer_idx,
        crate::injection::LoRAInjectionPoint::UpProj,
        up_base,
        up_base_id,
        x,
        x_id,
    )?;

    let dev = crate::pick_device_for_tensor(&gate_out);
    let (storage, _handle) = dev.silu_mul(
        gate_out.storage().as_ref(),
        up_out.storage().as_ref(),
        gate_out.shape(),
    )?;

    let swiglu_tensor = Tensor::new(
        Arc::from(storage),
        gate_out.shape().clone(),
        gate_out.dtype(),
        gate_out.provenance().clone(),
        gate_out.device().clone(),
    );

    let swiglu_id = tape.record_silu_mul(gate_id, up_id, swiglu_tensor.clone());
    Ok((swiglu_id, swiglu_tensor))
}

/// Arguments for FakeQuantInt4 (INT4 fake-quantization with STE) backward.
#[derive(Debug, Clone)]
pub struct FakeQuantInt4Args {
    /// Input tensor to fake-quantize (f32).
    pub input: Tensor,
    /// Input tensor ID on the tape.
    pub input_id: usize,
    /// Quantization scale (per-tensor or per-group).
    pub scale: f32,
    /// Zero point offset (default 0 for unsigned int4).
    pub zero_point: i32,
    /// Number of bits (always 4 for FakeQuantInt4).
    pub num_bits: u32,
}

/// Forward: fake-quantize `input` to int4 and dequantize back to f32.
///
/// The computation is:
/// 1. `q = clamp(round(input / scale) - zero_point, 0, 2^num_bits - 1)`
/// 2. `dequant = (q + zero_point) * scale`
///
/// The STE (Straight-Through Estimator) backward rule passes the
/// gradient of the dequantized output straight through to the
/// input, bypassing the non-differentiable clamp/round quantization.
/// This is the standard STE approach for QAT (Quantization-Aware
/// Training) with int4 fake quantization.
pub fn fake_quant_int4_forward(args: &FakeQuantInt4Args) -> Result<Tensor> {
    let data = args.input.to_vec_f32()?;
    let qmin: f32 = 0.0;
    let qmax: f32 = (1u32 << args.num_bits) as f32 - 1.0;

    let quantized: Vec<u8> = data
        .iter()
        .map(|&v| {
            let q = ((v / args.scale) - args.zero_point as f32)
                .round()
                .clamp(qmin, qmax);
            q as u8
        })
        .collect();

    let dequantized: Vec<f32> = quantized
        .iter()
        .map(|&q| (q as f32 + args.zero_point as f32) * args.scale)
        .collect();

    Ok(grim_backend_cpu::cpu_tensor(
        dequantized,
        args.input.shape().clone(),
    ))
}

/// Backward: STE — gradient passes through as identity (gradient of
/// quantize+dequant w.r.t. input is 1.0 everywhere in the STE
/// approximation, ignoring the non-differentiable clamp/round).
pub fn fake_quant_int4_backward(_args: &FakeQuantInt4Args, grad_output: &Tensor) -> Result<Tensor> {
    // STE: dx = grad_output * 1.0 (identity through quantize).
    let grad_data = grad_output.to_vec_f32()?;
    let out = grim_backend_cpu::cpu_tensor(grad_data, grad_output.shape().clone());
    Ok(out)
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
        use grim_tensor::{ArithType, DType, Device, QuantProvenance, Storage};

        let provenance = QuantProvenance::WithResiduals {
            outlier_count: 5,
            outlier_indices_offset: 1024,
            outlier_values_offset: 2048,
            outlier_indices: vec![],
            outlier_values_bits: vec![],
            primary_scale_offset: 0,
            primary_scale_size: 0,
            primary_row_scale_dtype: 0,
            primary_scale_bytes: vec![],
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

    /// Regression guard (WI-A): verifies that matmul_backward correctly extracts and uses explicit
    /// per-column scales for ResidualPacked and GroupInt tensors, and rejects missing scales.
    #[test]
    fn test_matmul_backward_residual_packed_scales_plumbing() {
        use grim_tensor::dtype::{
            ArithType, DType, Device, QuantProvenance, ResidualPackedConfig, Storage,
        };

        // Construct a ResidualPacked tensor with explicit non-unit scales.
        let storage = grim_backend_cpu::CpuStorage::new(
            vec![1.0f32; 4],
            Shape::new(vec![2, 2]),
            DType {
                arith: ArithType::F32,
                storage: Storage::ResidualPacked(ResidualPackedConfig { bpw: 4 }),
            },
        )
        .with_quant_scales(vec![2.5f32, 2.5f32]);

        let b_tensor = Tensor::new(
            Arc::new(storage),
            Shape::new(vec![2, 2]),
            DType {
                arith: ArithType::F32,
                storage: Storage::ResidualPacked(ResidualPackedConfig { bpw: 4 }),
            },
            QuantProvenance::GrimNative,
            Device::Cpu,
        );

        // Verify accessor extracts scales
        assert_eq!(b_tensor.quant_scales(), Some(&[2.5f32, 2.5f32][..]));

        // Construct un-scaled ResidualPacked tensor to verify missing scales return Error
        let unscaled_storage = grim_backend_cpu::CpuStorage::new(
            vec![1.0f32; 4],
            Shape::new(vec![2, 2]),
            DType {
                arith: ArithType::F32,
                storage: Storage::ResidualPacked(ResidualPackedConfig { bpw: 4 }),
            },
        );
        let b_unscaled_tensor = Tensor::new(
            Arc::new(unscaled_storage),
            Shape::new(vec![2, 2]),
            DType {
                arith: ArithType::F32,
                storage: Storage::ResidualPacked(ResidualPackedConfig { bpw: 4 }),
            },
            QuantProvenance::GrimNative,
            Device::Cpu,
        );

        let a = grim_backend_cpu::cpu_tensor(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2]));
        let out_grad =
            grim_backend_cpu::cpu_tensor(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2]));

        let args_unscaled = MatMulArgs {
            a: a.clone(),
            b: b_unscaled_tensor,
            out_grad: out_grad.clone(),
            transpose_a: false,
            transpose_b: false,
        };

        // matmul_backward must fail when required scales are missing
        assert!(matmul_backward(&args_unscaled).is_err());
    }

    #[test]
    fn test_apply_and_record_lora_direct() {
        use crate::{InjectionConfig, LoRAInjectionPoint, LoRAInjectionRegistry};
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

        let base_tensor =
            grim_backend_cpu::cpu_tensor(vec![1.0f32, 2.0f32], Shape::new(vec![1, 2]));
        let base_id = tape.register(base_tensor.clone());

        let x_tensor = grim_backend_cpu::cpu_tensor(vec![0.5f32, 0.5f32], Shape::new(vec![1, 2]));
        let x_id = tape.register(x_tensor.clone());

        let (_out_id, out_tensor) = apply_and_record_lora(
            &registry,
            &mut tape,
            0,
            LoRAInjectionPoint::QProj,
            base_tensor,
            base_id,
            x_tensor,
            x_id,
        )
        .unwrap();

        assert_eq!(out_tensor.shape().dims(), &[1, 2]);
        assert!(tape.len() > 0);
    }

    #[test]
    fn test_fused_swiglu_lora_mlp_direct() {
        use crate::{InjectionConfig, LoRAInjectionRegistry};
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

        let gate_base = grim_backend_cpu::cpu_tensor(
            vec![0.5f32, 1.0f32, 1.5f32, 2.0f32],
            Shape::new(vec![1, 4]),
        );
        let gate_id = tape.register(gate_base.clone());
        let up_base = grim_backend_cpu::cpu_tensor(
            vec![2.0f32, 3.0f32, 4.0f32, 5.0f32],
            Shape::new(vec![1, 4]),
        );
        let up_id = tape.register(up_base.clone());
        let x = grim_backend_cpu::cpu_tensor(vec![0.5f32, 0.5f32], Shape::new(vec![1, 2]));
        let x_id = tape.register(x.clone());

        let (_out_id, out_tensor) = fused_swiglu_lora_mlp(
            &registry, &mut tape, 0, gate_base, gate_id, up_base, up_id, x, x_id,
        )
        .unwrap();

        assert_eq!(out_tensor.shape().dims(), &[1, 4]);
        assert!(tape.len() > 0);
    }

    // --- FakeQuantInt4 (P5 Task 5.3) ---

    #[test]
    fn fake_quant_int4_round_trips_scale_1() {
        // With scale=1.0 and zero_point=0, values in [0,15] round-trip
        // exactly through quantize->dequantize.
        let input = tensor(vec![0.0, 7.5, 15.0, 3.0], vec![4]);
        let args = FakeQuantInt4Args {
            input,
            input_id: 0,
            scale: 1.0,
            zero_point: 0,
            num_bits: 4,
        };
        let out = fake_quant_int4_forward(&args).unwrap();
        let data = out.to_vec_f32().unwrap();
        // 0→0, 7.5→8 (round), 15→15, 3→3 (all within [0,15] so exact)
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 8.0);
        assert_eq!(data[2], 15.0);
        assert_eq!(data[3], 3.0);
    }

    #[test]
    fn fake_quant_int4_ste_backward_passes_gradient_through() {
        // STE backward: gradient passes through as identity.
        let grad_output = tensor(vec![1.0, -1.0, 0.5, 2.0], vec![4]);
        let args = FakeQuantInt4Args {
            input: tensor(vec![0.0, 0.0, 0.0, 0.0], vec![4]),
            input_id: 0,
            scale: 1.0,
            zero_point: 0,
            num_bits: 4,
        };
        let grad_input = fake_quant_int4_backward(&args, &grad_output).unwrap();
        assert_eq!(grad_input.to_vec_f32().unwrap(), vec![1.0, -1.0, 0.5, 2.0]);
    }

    // --- VeRA (Phase 9.3) ---

    #[test]
    fn vera_forward_outputs_quantized_low_rank_delta() {
        // x: [1, 2], a: [1, 2], b: [3, 1], rank=1, d_in=2, d_out=3.
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let a = tensor(vec![0.5, -1.0], vec![1, 2]);
        let b = tensor(vec![2.0, 1.0, -1.0], vec![3, 1]);

        // BA = B·A = [[1.0, -2.0], [0.5, -1.0], [-0.5, 1.0]].
        // Codebooks (single codebook of size 2): centroids {-2.0, 1.0}.
        let codebook = vec![vec![-2.0, 1.0]];

        let y = vera_forward(&x, &a, &b, &codebook, 1.0, 1).unwrap();
        let data = y.to_vec_f32().unwrap();
        // BA rows: [1,-2], [0.5,-1], [-0.5,1] → quantized with {-2,1}:
        //   row0 [1,-2] → [1,-2]; row1 [0.5,-1] → [1,-2]; row2 [-0.5,1] → [-2,1] (tie → first).
        // Y = x @ ĥBA^T, x=[1,2]: row0 1*1+2*(-2)=-3; row1 -3; row2 1*(-2)+2*1=0.
        assert_eq!(data, vec![-3.0, -3.0, 0.0]);
    }

    #[test]
    fn vera_forward_round_trips_exact_codebook_values() {
        // When BA values land exactly on centroids, output = exact scaled matmul.
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let a = tensor(vec![1.0, 0.0], vec![1, 2]);
        let b = tensor(vec![3.0], vec![1, 1]);
        // BA = [[3, 0]]; centroid set {3.0, 0.0} reproduces it exactly.
        let codebook = vec![vec![3.0, 0.0]];
        let y = vera_forward(&x, &a, &b, &codebook, 2.0, 1).unwrap();
        // Y = scale * x @ BA^T = 2 * (1*3 + 2*0) = 6.
        assert_eq!(y.to_vec_f32().unwrap(), vec![6.0]);
    }

    #[test]
    fn vera_backward_returns_five_gradients_with_st_estimator() {
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let a = tensor(vec![1.0, 0.0], vec![1, 2]);
        let b = tensor(vec![3.0], vec![1, 1]);
        let out_grad = tensor(vec![1.0], vec![1, 1]);
        let codebook = vec![vec![3.0, 0.0]];

        let (grad_base, grad_x, grad_a, grad_b, grad_cb) =
            vera_backward(&out_grad, &x, &a, &b, &codebook, 2.0).unwrap();

        // grad_base is a passthrough of out_grad.
        assert_eq!(grad_base.to_vec_f32().unwrap(), vec![1.0]);
        // ∇BA = scale * X^T @ G = 2 * [1, 2]^T * 1 = [2, 4].
        // ∇B = Σ_j ∇BA[j] * A[j] = 2*1 + 4*0 = 2.
        assert_eq!(grad_b.to_vec_f32().unwrap(), vec![2.0]);
        // ∇A[j] = B * ∇BA[j] = 3 * [2, 4] = [6, 12].
        assert_eq!(grad_a.to_vec_f32().unwrap(), vec![6.0, 12.0]);
        // ∇X = scale * G @ ĥBA = 2 * 1 * [3, 0] = [6, 0].
        assert_eq!(grad_x.to_vec_f32().unwrap(), vec![6.0, 0.0]);
        // Centroid 3.0 received ∇BA[0]=2, centroid 0.0 received ∇BA[1]=4.
        assert_eq!(grad_cb.to_vec_f32().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn vera_backward_rejects_ragged_codebooks() {
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let a = tensor(vec![1.0, 0.0], vec![1, 2]);
        let b = tensor(vec![3.0], vec![1, 1]);
        let out_grad = tensor(vec![1.0], vec![1, 1]);
        let codebook = vec![vec![3.0, 0.0], vec![1.0]];
        assert!(vera_backward(&out_grad, &x, &a, &b, &codebook, 1.0).is_err());
    }
}
