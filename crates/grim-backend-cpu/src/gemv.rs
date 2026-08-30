//! CPU GEMV matrix-vector multiplication kernel for expert offload.
//!
//! Provides host-RAM matrix-vector multiplication for computing offloaded MoE
//! expert feed-forward passes directly on CPU without GPU round-trips.

use grim_tensor::error::{Error, Result};

/// Multiplies row-major matrix `a` $[M, N]$ by vector `x` $[N]$, producing vector $y$ $[M]$.
///
/// # Mathematical Guarantee
/// $$y_i = \sum_{j=0}^{N-1} A_{i, j} \cdot x_j$$
pub fn cpu_gemv(a: &[f32], x: &[f32], m: usize, n: usize) -> Result<Vec<f32>> {
    if a.len() != m * n {
        return Err(Error::Backend(format!(
            "cpu_gemv: matrix size mismatch (expected {}, got {})",
            m * n,
            a.len()
        )));
    }
    if x.len() != n {
        return Err(Error::Backend(format!(
            "cpu_gemv: vector size mismatch (expected {n}, got {})",
            x.len()
        )));
    }

    let mut y = vec![0.0f32; m];
    for i in 0..m {
        let row_start = i * n;
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += a[row_start + j] * x[j];
        }
        y[i] = acc;
    }
    Ok(y)
}
