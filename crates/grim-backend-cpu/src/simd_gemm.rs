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
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let expected = vec![1.0, 2.0, 3.0, 4.0]; // identity
        
        let mut c = vec![0.0f32; 4];
        gemm_f32_simd(2, 2, 2, &a, &b, &mut c);
        
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_gemm_lora_fused() {
        let x = vec![1.0, 1.0]; // 1x2
        let w = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let a = vec![0.5, 0.5]; // 2x1
        let b = vec![1.0, 1.0]; // 1x2
        let scale = 1.0;
        
        let mut y = vec![0.0f32; 2];
        gemm_f32_lora_fused(1, 2, 2, 1, &x, &w, &a, &b, scale, &mut y);
        
        // Y = X*W + scale*(X*A*B) = [1,1] + scale*[1,1]*[1,1] = [2,2]
        assert!(y[0] > 1.9 && y[0] < 2.1);
        assert!(y[1] > 1.9 && y[1] < 2.1);
    }
}