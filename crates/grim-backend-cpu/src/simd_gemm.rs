//! SIMD-accelerated GEMM kernel for CPU backend.
//! 
//! Uses AVX2/SSE on x86_64 for fused matrix multiplication.
//! §4: OxiBLAS SIMD GEMM implementation.

use std::arch::x86_64::*;

/// Transpose a row-major matrix from `[rows, cols]` to `[cols, rows]`.
///
/// `mat` is the source laid out as `[rows, cols]` row-major; the returned
/// `Vec` is laid out as `[cols, rows]` row-major, i.e. element `(r, c)` of the
/// input lives at `out[c * rows + r]`.
fn transpose_row_major(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    out
}

/// SIMD GEMM: C = A * B^T
/// A: [M, K], B: [N, K], C: [M, N]
/// Uses AVX2 when available, falls back to scalar.
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
    // Scalar fallback
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
                
                // Process 8 elements at a time
                while kk + 8 <= k {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * k + kk));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(j * k + kk));
                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                    kk += 8;
                }
                
                // Horizontal sum of AVX2 register
                let sum_high = _mm256_extractf128_ps::<1>(sum);
                let sum_low = _mm256_castps256_ps128(sum);
                let sum = _mm_add_ps(sum_low, sum_high);
                
                let mut sum_arr = [0.0f32; 4];
                _mm_storeu_ps(sum_arr.as_mut_ptr(), sum);
                let mut total = sum_arr.iter().sum::<f32>();
                
                // Handle remaining elements
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

/// Batched GEMM for LoRA adapter fusion.
/// Computes Y = X * W + X * A * B for rank-decomposition adapters.
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
    // Compute X * W.
    // gemm_f32_simd reads its second argument as [N, K] row-major and returns
    // X * W^T, so the caller supplies W in [N, K] row-major.
    gemm_f32_simd(m, n, k, x, w, y);

    // Compute the LoRA term (X * A) * B and add it to the result.
    // Doc contract: A is [K, rank] row-major, B is [rank, N] row-major.
    // gemm_f32_simd computes C = LHS * RHS^T with RHS in [cols, K] row-major,
    // so to get (X * A) we must feed it A^T, i.e. A laid out as [rank, K]
    // row-major (a[r * k + h]).
    let mut intermediate = vec![0.0f32; m * lora_rank];
    let a_t = transpose_row_major(a, k, lora_rank); // [K, rank] -> [rank, K]
    gemm_f32_simd(m, lora_rank, k, x, &a_t, &mut intermediate);

    // Y += intermediate * B * scale. gemm_f32_simd would compute
    // intermediate * B^T and expects B in [N, rank] row-major; for the plain
    // (intermediate * B) product with B in [rank, N] row-major we do the
    // rank-reduction by hand instead.
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
        // M=2, N=3, K=4: non-square, non-symmetric matrices so that
        // A*B != A*B^T, exposing transpose bugs in B access.
        let a = vec![
            1.0, 2.0, 3.0, 4.0,  // row 0 of A (2x4)
            5.0, 6.0, 7.0, 8.0,  // row 1 of A
        ];
        // B stored row-major as N x K (3x4); GEMM computes C = A * B^T
        let b = vec![
            1.0,  2.0,  3.0,  4.0,   // row 0 of B (K=4)
            5.0,  6.0,  7.0,  8.0,   // row 1 of B
            9.0, 10.0, 11.0, 12.0,   // row 2 of B
        ];
        // Hand-computed expected C (2x3): C[i][j] = sum_k A[i][k] * B[j][k]
        let expected = vec![
            30.0,  70.0, 110.0,  // C[0][0..2]
            70.0, 174.0, 278.0,  // C[1][0..2]
        ];

        let mut c = vec![0.0f32; 6];
        gemm_f32_simd(2, 3, 4, &a, &b, &mut c);

        for i in 0..6 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_gemm_transposed_b() {
        // Verifies that the kernel correctly uses B^T (not B).
        // If the kernel mistakenly treats B as non-transposed, it would
        // compute A * B instead of A * B^T, producing wrong results for
        // non-symmetric B.
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
        // B is 2x4 (N=2, K=4), so B^T is 4x2
        let b = vec![
            1.0, 2.0, 3.0, 4.0,  // row 0 of B
            5.0, 6.0, 7.0, 8.0,  // row 1 of B
        ];
        // C[0][j] = sum_k A[0*K+k] * B[j*K+k]
        // C[0][0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
        // C[0][1] = 1*5 + 2*6 + 3*7 + 4*8 = 70
        let expected = vec![30.0, 70.0];

        let mut c = vec![0.0f32; 2];
        gemm_f32_simd(1, 2, 4, &a, &b, &mut c);

        for i in 0..2 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "transposed B mismatch at {}: got {} expected {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_gemm_lora_fused() {
        // Exercises the fused-LoRA path with lora_rank = 2 and non-symmetric
        // matrices so that A*B != A^T*B^T. This distinguishes three states:
        //   - the intended Y = X*W^T + scale*((X*A)*B)         -> [21, 46, 41]
        //   - the original yippee.md bug  Y = X*W^T + scale*(X*A^T*B)   -> [17, 40, 47]
        //   - the partial dccc5f6 fix   Y = X*W^T + scale*(X*A^T*B^T)  -> [37, 10, 57]
        // Only the correct implementation produces [21, 46, 41]; the test pins
        // that exact vector so any regression back to either transposed form
        // fails loudly.
        //
        // Contract: X is [M, K], W is [N, K] row-major (kernel computes X*W^T),
        // A is [K, rank] row-major, B is [rank, N] row-major.
        let m = 1;
        let n = 3;
        let k = 2;
        let lora_rank = 2;
        let x = vec![1.0, 2.0]; // [1, 2]
        let w = vec![
            1.0, 3.0, // row 0 of W (N=3, K=2)
            2.0, 4.0, // row 1 of W
            5.0, 6.0, // row 2 of W
        ];
        let a = vec![
            1.0, 2.0, // row 0 of A (K=2, rank=2)
            3.0, 1.0, // row 1 of A
        ];
        let b = vec![
            1.0, 2.0, 0.0, // row 0 of B (rank=2, N=3)
            0.0, 1.0, 3.0, // row 1 of B
        ];
        let scale = 2.0;

        // Hand-computed expected: Y = X*W^T + scale*((X*A)*B)
        // X*W^T (1x3): C[0][j] = sum_k X[k]*W[j*K+k]
        //   C[0] = 1*1 + 2*3 = 7
        //   C[1] = 1*2 + 2*4 = 10
        //   C[2] = 1*5 + 2*6 = 17
        // X*A (1x2): [1*1 + 2*3, 1*2 + 2*1] = [7, 4]
        // (X*A)*B (1x3): [7*1 + 4*0, 7*2 + 4*1, 7*0 + 4*3] = [7, 18, 12]
        // Y = [7+2*7, 10+2*18, 17+2*12] = [21, 46, 41]
        let expected = vec![21.0, 46.0, 41.0];

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
