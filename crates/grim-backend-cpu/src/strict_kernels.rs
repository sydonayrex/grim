//! Strict-mode deterministic kernels — §5.8.
//!
//! Three predicates committed to the architecture:
//! 1. **strict_matmul** — bit-stable reference matmul. Fixed-order scalar
//!    reduction; no SIMD auto-vectorization reordering; no FMA
//!    reassociation that breaks f32 determinism across compilers.
//! 2. **strict_softmax** — stable max-subtract, deterministic iteration
//!    order, no parallel reductions.
//! 3. **strict_attention** — score matrix constructed bottom-up with the
//!    same iteration order as `strict_matmul` + `strict_softmax`.
//!
//! These kernels are *intentionally* slower than non-strict variants and
//! the cost is exposed via `SIMD_DISABLED` so future phases can validate
//! that no one quietly replaces them with vectorized math. The
//! architecture explicitly says: "deterministic replay's cost should
//! stay visible, not get quietly optimized away".
//!
//! Hooking up to `DeterminismMode::Strict`: the engine bridges
//! `scheduler.determinism_mode` into the kernel selection. v1 routes
//! all strict-mode calls to these scalar kernels.

use crate::cpu_tensor;
use grim_tensor::Shape;

/// Strict matmul: `out[m, n] = sum_k a[m, k] * b[k, n]`.
///
/// Layout: `a` is `[M, K]` row-major, `b` is `[K, N]` row-major. Output
/// shape `[M, N]`. All reductions are scalar and in `(m, n, k)` order
/// so cross-platform results match.
pub fn strict_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Strict softmax: row-wise over the last dim. Stable by subtracting the
/// row max before `exp`. Iteration order is `(row, col)` to match the
/// configuration of other ops in the strict family.
pub fn strict_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        // 1. Row max (stable).
        let mut max = f32::NEG_INFINITY;
        for c in 0..cols {
            let v = input[off + c];
            if v > max {
                max = v;
            }
        }
        // 2. exp(x - max) accumulation.
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (input[off + c] - max).exp();
            out[off + c] = e;
            sum += e;
        }
        // 3. Normalize.
        for c in 0..cols {
            out[off + c] /= sum;
        }
    }
    out
}

/// Strict scaled dot-product attention. Input layout mirrors Llama's
/// decoding path: Q/K/V are `[tokens, num_heads * head_dim]`. KV-grouping
/// `num_kv_heads <= num_heads` is honoured in the same per-(query, key)
/// order as `strict_matmul` would produce.
#[allow(clippy::too_many_arguments)]
pub fn strict_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_tokens * num_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;
    for h in 0..num_heads {
        let kvh = (h * num_kv_heads) / num_heads.max(1);
        for qt in 0..num_tokens {
            // scores[kt] = (q · k[kt]) * scale
            let mut scores = vec![0.0f32; num_tokens];
            for kt in 0..num_tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qt * num_heads * head_dim + h * head_dim + d]
                        * k[kt * kv_stride + kvh * head_dim + d];
                }
                scores[kt] = dot * scale;
            }
            // Stable softmax in-place on `scores`.
            let mut max = f32::NEG_INFINITY;
            for kt in 0..num_tokens {
                if scores[kt] > max {
                    max = scores[kt];
                }
            }
            let mut sum = 0.0f32;
            for kt in 0..num_tokens {
                scores[kt] = (scores[kt] - max).exp();
                sum += scores[kt];
            }
            for kt in 0..num_tokens {
                scores[kt] /= sum;
            }
            // Weighted sum against V.
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for kt in 0..num_tokens {
                    acc += scores[kt] * v[kt * kv_stride + kvh * head_dim + d];
                }
                out[qt * num_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }
    out
}

/// Convenience: matmul that returns a `Tensor` with the `[M, N]` shape.
pub fn strict_matmul_tensor(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> grim_tensor::Tensor {
    let v = strict_matmul(a, b, m, k, n);
    cpu_tensor(v, Shape::new(vec![m, n]))
}

/// Convenience: softmax that returns a `Tensor` with the same shape as
/// the input (`[rows, cols]`).
pub fn strict_softmax_tensor(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> grim_tensor::Tensor {
    let v = strict_softmax(input, rows, cols);
    cpu_tensor(v, Shape::new(vec![rows, cols]))
}

/// Convenience: attention that returns a `Tensor` with the
/// `[num_tokens, num_heads, head_dim]` shape.
#[allow(clippy::too_many_arguments)]
pub fn strict_attention_tensor(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> grim_tensor::Tensor {
    let v = strict_attention(q, k, v, num_tokens, num_heads, num_kv_heads, head_dim);
    cpu_tensor(v, Shape::new(vec![num_tokens, num_heads, head_dim]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity_left_half_matches_strict() {
        // Test: a = identity, b = arbitrary → out = b.
        let m = 4usize;
        let k = 4usize;
        let n = 3usize;
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            a[i * k + i] = 1.0;
        }
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = strict_matmul(&a, &b, m, k, n);
        // Row i should equal b[i*n..(i+1)*n].
        for i in 0..m {
            for j in 0..n {
                assert_eq!(out[i * n + j], b[i * n + j]);
            }
        }
    }

    #[test]
    fn matmul_is_deterministic_under_repeated_calls() {
        // Two identical inputs must produce bit-identical outputs across
        // calls — that's the strict-mode reproducibility contract.
        let a = vec![1.0f32; 12];
        let b = vec![0.5f32; 12];
        let o1 = strict_matmul(&a, &b, 3, 4, 3);
        let o2 = strict_matmul(&a, &b, 3, 4, 3);
        let mut diff_bits = 0u64;
        for (x, y) in o1.iter().zip(o2.iter()) {
            let x_bits = x.to_bits();
            let y_bits = y.to_bits();
            diff_bits |= (x_bits as u64) ^ (y_bits as u64);
        }
        assert_eq!(diff_bits, 0, "strict_matmul must be bit-exact across runs");
    }

    #[test]
    fn matmul_handles_multiple_reproducible_runs() {
        // Three independent calls with the same inputs must produce
        // the same output, validating strictness in a broader sample.
        let a: Vec<f32> = (0..30).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..30).map(|i| (i as f32 * 0.07).cos()).collect();
        let r1 = strict_matmul(&a, &b, 5, 6, 5);
        let r2 = strict_matmul(&a, &b, 5, 6, 5);
        let r3 = strict_matmul(&a, &b, 5, 6, 5);
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
    }

    #[test]
    fn softmax_outputs_sum_to_one_per_row() {
        let rows = 3usize;
        let cols = 5usize;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.7).collect();
        let out = strict_softmax(&input, rows, cols);
        for r in 0..rows {
            let s: f32 = (0..cols).map(|c| out[r * cols + c]).sum();
            assert!((s - 1.0).abs() < 1e-5, "row {r} sum should be ~1.0, got {s}");
        }
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        // Numerical-stability check: softmax of large positives must
        // not over/underflow.
        let input = vec![1.0e6, 1.0e6 + 1.0, 1.0e6 + 2.0, 1.0e6 + 3.0];
        let out = strict_softmax(&input, 1, 4);
        for v in &out {
            assert!(v.is_finite(), "softmax output {v} is non-finite");
        }
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn attention_self_loop_returns_softmaxed_qkt() {
        // Q = K = V = ones on a single token; result should be 1.0
        // everywhere because softmax(num_heads, num_kv_heads) reduces
        // to ones (the score matrix is uniform).
        let num_tokens = 1usize;
        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;
        let q = vec![1.0f32; num_heads * head_dim];
        let k = vec![1.0f32; num_kv_heads * head_dim];
        let v = vec![1.0f32; num_kv_heads * head_dim];
        let out = strict_attention(&q, &k, &v, num_tokens, num_heads, num_kv_heads, head_dim);
        for v in &out {
            // Scale is 1.0 / sqrt(4) = 0.5; dot = 4; softmax(0) = 1.0; out = 1.0.
            assert!((v - 1.0).abs() < 1e-5, "expected ~1.0, got {v}");
        }
    }
}
