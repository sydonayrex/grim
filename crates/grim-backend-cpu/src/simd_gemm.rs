//! SIMD GEMM kernel for CPU backend (AVX2/SSE on x86_64). §4: OxiBLAS SIMD GEMM.

use std::arch::x86_64::*;

/// Transpose `[rows, cols]` to `[cols, rows]` row-major.
fn transpose_row_major(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    out
}

/// `C = A @ B^T`, all row-major. AVX2 when available; scalar fallback.
pub fn gemm_f32_simd(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                gemm_f32_avx2(m, n, k, a, b, c);
            }
            return;
        }
    }
    // Scalar fallback.
    gemm_f32_scalar(m, n, k, a, b, c);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemm_f32_avx2(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut sum = _mm256_setzero_ps();
                let mut kk = 0;

                // Process 8 elements at a time (AVX2).
                while kk + 8 <= k {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * k + kk));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(j * k + kk));
                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                    kk += 8;
                }

                // Horizontal sum of AVX2 register.
                let sum_high = _mm256_extractf128_ps::<1>(sum);
                let sum_low = _mm256_castps256_ps128(sum);
                let sum = _mm_add_ps(sum_low, sum_high);

                let mut sum_arr = [0.0f32; 4];
                _mm_storeu_ps(sum_arr.as_mut_ptr(), sum);
                let mut total = sum_arr.iter().sum::<f32>();

                // Handle remaining elements.
                for kk_rem in kk..k {
                    total += a[i * k + kk_rem] * b[j * k + kk_rem];
                }
                c[i * n + j] = total;
            }
        }
    }
}

fn gemm_f32_scalar(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[j * k + kk];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Fused LoRA GEMM: `Y = X*W^T + scale*((X*A)*B)`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f32_lora_fused(
    m: usize,
    n: usize,
    k: usize,
    lora_rank: usize,
    x: &[f32],
    w: &[f32],
    a: &[f32],
    b: &[f32],
    scale: f32,
    y: &mut [f32],
) {
    gemm_f32_simd(m, n, k, x, w, y);

    // LoRA addition: (X * A) * B. gemm_f32_simd computes LHS * RHS^T → feed A^T.
    let mut intermediate = vec![0.0f32; m * lora_rank];
    let a_t = transpose_row_major(a, k, lora_rank);
    gemm_f32_simd(m, lora_rank, k, x, &a_t, &mut intermediate);

    // Y += intermediate * B * scale (rank-reduction loop).
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for r in 0..lora_rank {
                sum += intermediate[i * lora_rank + r] * b[r * n + j];
            }
            y[i * n + j] += sum * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_basic() {
        // Non-square, non-symmetric: exposes A*B vs A*B^T mismatch.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        // C[i][j] = sum_k A[i][k] * B[j][k].
        let expected = [
            30.0, 70.0, 110.0, // C[0][0..2]
            70.0, 174.0, 278.0, // C[1][0..2]
        ];

        let mut c = [0.0f32; 6];
        gemm_f32_simd(2, 3, 4, &a, &b, &mut c);

        for i in 0..6 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i,
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_gemm_transposed_b() {
        // Verifies kernel computes A * B^T, not A * B.
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
        // B is 2x4 (N=2, K=4), so B^T is 4x2.
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // C[0][j] = sum_k A[0*K+k] * B[j*K+k]
        // C[0][0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
        // C[0][1] = 1*5 + 2*6 + 3*7 + 4*8 = 70
        let expected = [30.0, 70.0];

        let mut c = vec![0.0f32; 2];
        gemm_f32_simd(1, 2, 4, &a, &b, &mut c);

        for i in 0..2 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "transposed B mismatch at {}: got {} expected {}",
                i,
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_gemm_lora_fused() {
        // Three states: (a) correct Y → [21,46,41], (b) bug: X*A^T*B → [17,40,47],
        // (c) bug: X*A^T*B^T → [37,10,57].
        let m = 1;
        let n = 3;
        let k = 2;
        let lora_rank = 2;
        let x = vec![1.0, 2.0]; // [1, 2]
        let w = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
        let a = vec![1.0, 2.0, 3.0, 1.0];
        let b = vec![1.0, 2.0, 0.0, 0.0, 1.0, 3.0];
        let scale = 2.0;

        // Hand-computed: X*W^T=[7,10,17]; X*A=[7,4]; (X*A)*B=[7,18,12]; Y=[21,46,41].
        let expected = [21.0, 46.0, 41.0];

        let mut y = vec![0.0f32; 3];
        gemm_f32_lora_fused(m, n, k, lora_rank, &x, &w, &a, &b, scale, &mut y);

        for i in 0..3 {
            assert!(
                (y[i] - expected[i]).abs() < 1e-5,
                "lora fused mismatch at {}: got {} expected {}",
                i,
                y[i],
                expected[i]
            );
        }
    }
}
