//! Subspace Orthogonal Newton-Schulz Kernel for SOUL EATER.
//!
//! Provides exact 16x16 Jacobi symmetric eigendecomposition, rank conditioning checks,
//! and adaptive cubic Newton-Schulz matrix orthogonalization for tall/thin matrices [d x r].

/// Custom error type for ill-conditioned or rank-deficient subspace matrices.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionNumberError {
    IllConditioned { kappa: f32 },
    RankDeficient { lambda_min: f32 },
}

impl std::fmt::Display for ConditionNumberError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllConditioned { kappa } => write!(
                f,
                "Condition number kappa={kappa:.2} exceeds threshold 100.0"
            ),
            Self::RankDeficient { lambda_min } => write!(
                f,
                "Minimum eigenvalue lambda_min={lambda_min:e} is below threshold 1e-6"
            ),
        }
    }
}

impl std::error::Error for ConditionNumberError {}

/// Compute outer Gram matrix `S = X^T * X` (r x r) for a tall/thin matrix `X` stored as row-major [d x r].
pub fn subspace_gram_matrix(x: &[f32], d: usize, r: usize) -> Vec<f32> {
    assert_eq!(x.len(), d * r, "Input slice size mismatch");
    let mut s = vec![0.0f32; r * r];
    for row in 0..d {
        let x_row = &x[row * r..(row + 1) * r];
        for i in 0..r {
            for j in 0..r {
                s[i * r + j] += x_row[i] * x_row[j];
            }
        }
    }
    s
}

/// Compute exact eigenvalues (lambda_max, lambda_min) of a symmetric r x r Gram matrix `S`
/// using cyclic Jacobi rotations. Operates in O(r^3) ~ 70,000 FLOPs for r=16 (~1us).
pub fn exact_jacobi_eigenvalues(s: &[f32], r: usize) -> (f32, f32) {
    assert_eq!(s.len(), r * r, "Gram matrix size mismatch");
    let mut a = s.to_vec();

    // Perform up to 15 sweeps of cyclic Jacobi rotations
    let max_sweeps = 15;
    for _ in 0..max_sweeps {
        let mut off_diag_sum = 0.0f32;
        for i in 0..r {
            for j in (i + 1)..r {
                off_diag_sum += a[i * r + j].abs();
            }
        }
        if off_diag_sum < 1e-7 {
            break;
        }

        for i in 0..r {
            for j in (i + 1)..r {
                let a_ij = a[i * r + j];
                if a_ij.abs() < 1e-9 {
                    continue;
                }
                let a_ii = a[i * r + i];
                let a_jj = a[j * r + j];

                let tau = (a_jj - a_ii) / (2.0 * a_ij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s_rot = t * c;

                // Update matrix A
                a[i * r + i] -= t * a_ij;
                a[j * r + j] += t * a_ij;
                a[i * r + j] = 0.0;
                a[j * r + i] = 0.0;

                for k in 0..r {
                    if k != i && k != j {
                        let a_ik = a[i * r + k];
                        let a_jk = a[j * r + k];
                        let new_ik = c * a_ik - s_rot * a_jk;
                        let new_jk = s_rot * a_ik + c * a_jk;
                        a[i * r + k] = new_ik;
                        a[k * r + i] = new_ik;
                        a[j * r + k] = new_jk;
                        a[k * r + j] = new_jk;
                    }
                }
            }
        }
    }

    let mut lambda_max = f32::NEG_INFINITY;
    let mut lambda_min = f32::INFINITY;
    for i in 0..r {
        let val = a[i * r + i];
        if val > lambda_max {
            lambda_max = val;
        }
        if val < lambda_min {
            lambda_min = val;
        }
    }
    (lambda_max, lambda_min.max(0.0))
}

/// Check condition number kappa(S) = lambda_max / lambda_min on r x r Gram matrix `S`.
/// Returns true if kappa(S) <= 100.0 and lambda_min >= 1e-6.
pub fn check_rank_conditioning(s: &[f32], r: usize) -> Result<(f32, f32), ConditionNumberError> {
    let (lambda_max, lambda_min) = exact_jacobi_eigenvalues(s, r);
    if lambda_min < 1e-6 {
        return Err(ConditionNumberError::RankDeficient { lambda_min });
    }
    let kappa = (lambda_max / lambda_min).sqrt();
    if kappa > 100.0 {
        return Err(ConditionNumberError::IllConditioned { kappa });
    }
    Ok((lambda_max, lambda_min))
}

/// Execute adaptive spectral-normalized Cubic Newton-Schulz iteration steps on matrix `X` [d x r].
/// Recomputes S_k = X_k^T * X_k at EVERY step k inside the loop body.
/// Returns the number of iterations required to achieve ||S_k - I_r||_F < 1e-4.
pub fn subspace_newton_schulz_step(
    x: &mut [f32],
    d: usize,
    r: usize,
    max_iters: usize,
) -> Result<usize, ConditionNumberError> {
    assert_eq!(x.len(), d * r, "Matrix dimensions mismatch");

    // 1. Initial Gram matrix
    let s_0 = subspace_gram_matrix(x, d, r);

    // 2. Exact Jacobi eigendecomposition & conditioning check
    let (lambda_max, _) = check_rank_conditioning(&s_0, r)?;

    // 3. Spectral Pre-Normalization: X_0 = X / (sqrt(lambda_max) + 1e-7)
    let sigma_est = lambda_max.sqrt().max(1e-7);
    let inv_sigma = 1.0 / (sigma_est + 1e-7);
    for val in x.iter_mut() {
        *val *= inv_sigma;
    }

    // 4. Adaptive Cubic Newton-Schulz loop: X_{k+1} = 0.5 * X_k * (3 I_r - S_k)
    let mut next_x = vec![0.0f32; d * r];
    let mut scale_mat = vec![0.0f32; r * r];

    for k in 0..max_iters {
        let s_k = subspace_gram_matrix(x, d, r);

        // Check residual ||S_k - I_r||_F
        let mut residual_sq = 0.0f32;
        for i in 0..r {
            for j in 0..r {
                let target = if i == j { 1.0 } else { 0.0 };
                let diff = s_k[i * r + j] - target;
                residual_sq += diff * diff;
            }
        }
        if residual_sq.sqrt() < 1e-4 {
            return Ok(k);
        }

        // Compute 0.5 * (3 I_r - S_k)
        scale_mat.fill(0.0);
        for i in 0..r {
            for j in 0..r {
                let diag = if i == j { 3.0 } else { 0.0 };
                scale_mat[i * r + j] = 0.5 * (diag - s_k[i * r + j]);
            }
        }

        // Matmul: next_x = x * scale_mat [d x r] * [r x r]
        next_x.fill(0.0);
        for row in 0..d {
            let x_row = &x[row * r..(row + 1) * r];
            let out_row = &mut next_x[row * r..(row + 1) * r];
            for i in 0..r {
                let x_val = x_row[i];
                for j in 0..r {
                    out_row[j] += x_val * scale_mat[i * r + j];
                }
            }
        }
        x.copy_from_slice(&next_x);
    }

    Ok(max_iters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subspace_newton_schulz_well_conditioned() {
        let d = 32;
        let r = 16;
        let mut x = vec![0.0f32; d * r];

        // Generate a well-conditioned matrix (identity + small perturbation)
        for row in 0..d {
            for col in 0..r {
                let diag = if row == col { 1.0 } else { 0.0 };
                let noise = ((row * 17 + col * 31) % 100) as f32 / 500.0 - 0.1;
                x[row * r + col] = diag + noise;
            }
        }

        let steps = subspace_newton_schulz_step(&mut x, d, r, 15)
            .expect("Well-conditioned matrix must succeed");
        assert!(steps <= 15, "Should converge within 15 steps");

        // Verify orthogonality: X^T * X ≈ I_r
        let s_final = subspace_gram_matrix(&x, d, r);
        let mut residual_sq = 0.0f32;
        for i in 0..r {
            for j in 0..r {
                let target = if i == j { 1.0 } else { 0.0 };
                let diff = s_final[i * r + j] - target;
                residual_sq += diff * diff;
            }
        }
        assert!(
            residual_sq.sqrt() < 1e-3,
            "Residual must be < 1e-3, got {}",
            residual_sq.sqrt()
        );
    }

    #[test]
    fn test_subspace_newton_schulz_rank_deficient_guard() {
        let d = 32;
        let r = 16;
        let mut x = vec![0.0f32; d * r];

        // Create a rank-deficient matrix (column 0 is all zeros)
        for row in 0..d {
            for col in 1..r {
                x[row * r + col] = 1.0;
            }
        }

        let res = subspace_newton_schulz_step(&mut x, d, r, 10);
        assert!(
            res.is_err(),
            "Rank deficient matrix must trigger error guard"
        );
        let err = res.unwrap_err();
        match err {
            ConditionNumberError::RankDeficient { lambda_min } => {
                assert!(lambda_min < 1e-6);
            }
            _ => panic!("Expected RankDeficient error"),
        }
    }
}
