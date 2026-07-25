//! SIMD-accelerated GEMM kernel for CPU backend.
//! 
//! Uses AVX2/SSE on x86_64 for fused matrix multiplication.
//! §4: OxiBLAS SIMD GEMM implementation.

use std::arch::x86_64::*;

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
    // Compute X * W
    gemm_f32_simd(m, n, k, x, w, y);
    
    // Compute X * A * B and add to result
    // A: [K, rank], B: [rank, N]
    // intermediate: [M, rank] = X * A
    let mut intermediate = vec![0.0f32; m * lora_rank];
    gemm_f32_simd(m, lora_rank, k, x, a, &mut intermediate);
    
    // Y += intermediate * B * scale
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for r in 0..lora_rank {
                sum += intermediate[i * lora_rank + r] * b[j * lora_rank + r];
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
        // Non-identity matrices with exact expected values.
        // GEMM convention: C = A * B^T (B stored as N x K row-major).
        // X: MxK = 1x2, W: NxK = 3x2, A: Kxrank = 2x1, B: rankxN = 1x3
        let x = vec![1.0, 2.0]; // 1x2 (M=1, K=2)
        let w = vec![
            1.0, 2.0,  // row 0 of W (N=3, K=2)
            3.0, 4.0,  // row 1 of W
            5.0, 6.0,  // row 2 of W
        ];
        let a = vec![0.5, 1.0]; // 2x1 (K x lora_rank)
        let b = vec![1.0, 2.0, 3.0]; // 1x3 (rank x N)
        let scale = 2.0;

        // Hand-computed expected: Y = X*W + scale*(X*A*B)
        // X*W (1x3), computing C[i][j] = sum_k X[k] * W[j*K+k]
        //   C[0] = X*W[0] = 1*1 + 2*2 = 5
        //   C[1] = X*W[1] = 1*3 + 2*4 = 11
        //   C[2] = X*W[2] = 1*5 + 2*6 = 17
        // X*A (1x1): [1*0.5 + 2*1.0] = [2.5]
        // X*A*B (1x3): [2.5*1, 2.5*2, 2.5*3] = [2.5, 5.0, 7.5]
        // Y = [5 + 2*2.5, 11 + 2*5.0, 17 + 2*7.5] = [10.0, 21.0, 32.0]
        let expected = vec![10.0, 21.0, 32.0];

        let mut y = vec![0.0f32; 3];
        gemm_f32_lora_fused(1, 3, 2, 1, &x, &w, &a, &b, scale, &mut y);

        for i in 0..3 {
            assert!(
                (y[i] - expected[i]).abs() < 1e-5,
                "lora fused mismatch at {}: got {} expected {}",
                i, y[i], expected[i]
            );
        }
    }
}
