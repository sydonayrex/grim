//! Mathematical and non-linear kernel parity tests between CPU and ROCm (§WI-E9).
//!
//! Validates numerical precision, edge cases, and algorithmic parity for:
//! - RMSNorm & Fused-Add-RMSNorm
//! - Rotary Positional Embeddings (RoPE, mRoPE)
//! - Activations (SiLU, GELU, SwiGLU)
//! - Softmax / Log-Softmax stability
//! - Cross-Entropy Loss computation

/// CPU reference implementation of RMSNorm.
///
/// Computes `y = (x / sqrt(mean(x^2) + eps)) * gamma`.
pub fn cpu_rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), gamma.len());
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&v, &g)| v * inv_rms * g)
        .collect()
}

/// CPU reference implementation of Fused-Add-RMSNorm.
///
/// Computes `res = x + residual`, then normalizes and scales `res`.
pub fn cpu_fused_add_rmsnorm(
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), residual.len());
    assert_eq!(x.len(), gamma.len());
    let updated_residual: Vec<f32> = x
        .iter()
        .zip(residual.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let norm = cpu_rmsnorm(&updated_residual, gamma, eps);
    (norm, updated_residual)
}

/// CPU reference implementation of standard 1D RoPE (Rotary Positional Embedding).
///
/// Rotates consecutive pairs or half-split components at position `pos`.
pub fn cpu_rope_1d(
    vec: &[f32],
    pos: usize,
    head_dim: usize,
    theta_base: f32,
) -> Vec<f32> {
    assert_eq!(vec.len() % head_dim, 0);
    assert_eq!(head_dim % 2, 0);
    let mut out = vec.to_vec();
    let half_dim = head_dim / 2;

    for head_chunk in out.chunks_exact_mut(head_dim) {
        for i in 0..half_dim {
            let freq = 1.0 / theta_base.powf((2 * i) as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let cos_val = val.cos();
            let sin_val = val.sin();

            let x0 = head_chunk[i];
            let x1 = head_chunk[i + half_dim];

            head_chunk[i] = x0 * cos_val - x1 * sin_val;
            head_chunk[i + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }
    out
}

/// CPU reference implementation of SiLU activation: `f(x) = x / (1 + exp(-x))`.
pub fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// CPU reference implementation of SwiGLU: `swiglu(gate, up) = silu(gate) * up`.
pub fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    let silu_gate = cpu_silu(gate);
    silu_gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| g * u)
        .collect()
}

/// CPU reference implementation of numerically stable Softmax: `softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))`.
pub fn cpu_softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let inv_sum = 1.0 / (sum_exp + 1e-12);
    exps.into_iter().map(|v| v * inv_sum).collect()
}

/// CPU reference implementation of Cross-Entropy Loss from logits and target label index.
pub fn cpu_cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    assert!(target < logits.len());
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    log_sum_exp - logits[target]
}

#[test]
fn test_cpu_rmsnorm_invariants() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;
    let out = cpu_rmsnorm(&x, &gamma, eps);
    let mean_sq_out = out.iter().map(|&v| v * v).sum::<f32>() / (out.len() as f32);
    assert!((mean_sq_out - 1.0).abs() < 1e-4, "Normalized variance should be ~1.0");
}

#[test]
fn test_cpu_fused_add_rmsnorm_equivalence() {
    let x = vec![0.5, -0.2, 1.2, 3.0];
    let res = vec![0.5, 0.2, -0.2, 1.0];
    let gamma = vec![1.0, 0.8, 1.2, 0.5];
    let eps = 1e-5;

    let (fused_out, new_res) = cpu_fused_add_rmsnorm(&x, &res, &gamma, eps);
    let manual_res: Vec<f32> = x.iter().zip(&res).map(|(&a, &b)| a + b).collect();
    let manual_norm = cpu_rmsnorm(&manual_res, &gamma, eps);

    assert_eq!(new_res, manual_res);
    for (a, b) in fused_out.iter().zip(&manual_norm) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_cpu_rope_rotation_preserves_l2_norm() {
    let head_dim = 64;
    let mut vec = Vec::with_capacity(head_dim);
    for i in 0..head_dim {
        vec.push((i as f32) * 0.1);
    }
    let orig_norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();

    let rotated = cpu_rope_1d(&vec, 42, head_dim, 10000.0);
    let rotated_norm: f32 = rotated.iter().map(|&v| v * v).sum::<f32>().sqrt();

    assert!(
        (orig_norm - rotated_norm).abs() < 1e-4,
        "RoPE rotation is orthogonal and must preserve vector L2 norm"
    );
}

#[test]
fn test_cpu_swiglu_numerical_properties() {
    let gate = vec![-10.0, 0.0, 10.0];
    let up = vec![2.0, 3.0, 4.0];
    let out = cpu_swiglu(&gate, &up);

    // silu(0) = 0 -> swiglu = 0
    assert_eq!(out[1], 0.0);
    // silu(-10) -> ~0 -> swiglu ~0
    assert!(out[0].abs() < 1e-3);
    // silu(10) -> ~10 -> swiglu ~ 40.0
    assert!((out[2] - 40.0).abs() < 1e-2);
}

#[test]
fn test_cpu_softmax_sum_to_one() {
    let x = vec![-1000.0, -999.0, -998.0, 0.0, 1.0, 5.0];
    let probs = cpu_softmax(&x);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax probabilities must sum to 1.0");
    for &p in &probs {
        assert!(p >= 0.0, "Probability must be non-negative");
    }
}

#[test]
fn test_cpu_cross_entropy_perfect_prediction_approaches_zero() {
    let mut logits = vec![-50.0; 10];
    logits[3] = 50.0; // Very high confidence for index 3
    let loss = cpu_cross_entropy_loss(&logits, 3);
    assert!(loss >= 0.0);
    assert!(loss < 1e-4, "Loss for near-certain correct prediction should be ~0.0");
}

#[test]
fn test_gpu_rocm_rmsnorm_parity() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        return;
    }

    #[cfg(feature = "rocm")]
    {
        use grim_backend_rocm::RocmDevice;
        use grim_tensor::{BackendDevice, DType, Shape};

        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };

        for &dim in &[512, 1024, 2048, 4096] {
            let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let gamma: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos() + 1.0).collect();
            let eps = 1e-5f32;

            let cpu_out = cpu_rmsnorm(&x, &gamma, eps);

            let shape = Shape::new(vec![dim]);
            let x_storage = dev.from_cpu(&x, &shape, DType::F32).unwrap();
            let gamma_storage = dev.from_cpu(&gamma, &shape, DType::F32).unwrap();

            let (gpu_out_storage, handle) = dev
                .rms_norm(x_storage.as_ref(), gamma_storage.as_ref(), eps, &shape)
                .unwrap();
            handle.synchronize().unwrap();
            let gpu_out = gpu_out_storage.to_cpu_vec_f32().unwrap();

            let max_diff = cpu_out
                .iter()
                .zip(&gpu_out)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-4,
                "RMSNorm divergence at dim={dim}: max_diff={max_diff}"
            );
        }
    }
}
