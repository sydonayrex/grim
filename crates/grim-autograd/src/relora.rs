//! ReLoRA: periodic merge of low-rank adapter delta into base weights,
//! zero the adapters, and restart optimizer momentum for those parameters.

/// Merge `scale * (b @ a)` into `base`, then zero `a` and `b`.
///
/// Shape contracts:
/// - `a`: `[rank, in_features]` row-major.
/// - `b`: `[out_features, rank]` row-major.
/// - `base`: `[out_features, in_features]` row-major.
pub fn merge_and_zero(
    rank: usize,
    in_features: usize,
    out_features: usize,
    scale: f32,
    a: &mut [f32],
    b: &mut [f32],
    base: &mut [f32],
) {
    assert_eq!(a.len(), rank * in_features, "a shape mismatch");
    assert_eq!(b.len(), out_features * rank, "b shape mismatch");
    assert_eq!(base.len(), out_features * in_features, "base shape mismatch");

    for i in 0..out_features {
        for j in 0..in_features {
            let mut s = 0.0f32;
            for k in 0..rank {
                // b[i, k] * a[k, j]
                s += b[i * rank + k] * a[k * in_features + j];
            }
            base[i * in_features + j] += scale * s;
        }
    }
    for v in a.iter_mut() {
        *v = 0.0;
    }
    for v in b.iter_mut() {
        *v = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_folds_delta_into_base_and_zeros_adapters() {
        let rank = 2;
        let in_features = 2;
        let out_features = 2;
        let scale = 0.5f32;

        let mut a = vec![0.1f32; rank * in_features];
        let mut b = vec![0.1f32; out_features * rank];
        let mut base = vec![1.0f32; out_features * in_features];

        merge_and_zero(rank, in_features, out_features, scale, &mut a, &mut b, &mut base);

        // Expected merged base: 1.0 + 0.5 * (0.1*0.1 + 0.1*0.1) = 1.0 + 0.5 * 0.02 = 1.01
        for val in &base {
            assert!((val - 1.01).abs() < 1e-6, "expected 1.01, got {val}");
        }
        for val in &a {
            assert_eq!(*val, 0.0, "expected a to be zeroed");
        }
        for val in &b {
            assert_eq!(*val, 0.0, "expected b to be zeroed");
        }
    }
}
