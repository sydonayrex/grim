//! SCALE: Stateless Column-wise RMS Gradient Normalization.
//!
//! Repurposed from SCALE (arXiv:2506.16659).
//! Replaces tracking per-element second moment EMA states (such as Adam's v or Fisher Information)
//! with instantaneous column-wise root-mean-square normalization across the weight matrix dimension.

/// Column-wise RMS normalization of a flat [d, r] gradient (row-major).
/// Returns g_col_norm[k] = sqrt(mean_j g[j*r+k]^2) + eps, length r.
pub fn column_rms(g: &[f32], d: usize, r: usize, eps: f32) -> Vec<f32> {
    if d == 0 || r == 0 {
        return vec![eps; r];
    }
    let mut sums = vec![0.0f32; r];
    for j in 0..d {
        for k in 0..r {
            let v = g[j * r + k];
            sums[k] += v * v;
        }
    }
    sums.iter()
        .map(|&s| (s / (d as f32)).sqrt() + eps)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_rms_1d_vector() {
        let g = vec![3.0f32, 4.0f32];
        let norms = column_rms(&g, 1, 2, 0.0);
        assert_eq!(norms, vec![3.0, 4.0]);
    }

    #[test]
    fn test_column_rms_2d_matrix() {
        // g = [[1, 2], [3, 4]], d = 2, r = 2
        // col0: (1^2 + 3^2) / 2 = 5 -> sqrt(5) ≈ 2.2360679
        // col1: (2^2 + 4^2) / 2 = 10 -> sqrt(10) ≈ 3.1622777
        let g = vec![1.0f32, 2.0, 3.0, 4.0];
        let norms = column_rms(&g, 2, 2, 0.0);

        let expected_col0 = (5.0f32).sqrt();
        let expected_col1 = (10.0f32).sqrt();

        assert!((norms[0] - expected_col0).abs() < 1e-6);
        assert!((norms[1] - expected_col1).abs() < 1e-6);
    }
}
