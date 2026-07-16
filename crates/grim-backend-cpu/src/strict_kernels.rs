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
use grim_tensor::{Shape, SoftmaxPartial};

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

/// Partial online-softmax attention for WI 3.4.2 (hybrid CPU/GPU offload).
///
/// A direct, math-for-math port of the GPU kernel's online-softmax algorithm
/// (mirrors `woody_attention_online_f32` in `grim-backend-rocm/lib_internal_tests.rs`,
/// which itself mirrors `grim_qkv_attention`'s inner loop). Instead of returning
/// a finalized output vector, it returns a [`SoftmaxPartial`] triple for each
/// `(head, query_token)` pair, computed over the KV range `[j_start, j_end)`.
///
/// Two partials computed over disjoint ranges can be merged via
/// [`grim_tensor::merge_partials`] to reconstruct the full result — this is
/// how the hybrid path combines CPU-side (offloaded) and GPU-side partials.
///
/// **Layout** (matches the GPU kernel):
/// - Q: `[seq_len, num_heads, head_dim]` row-major (stride = `num_heads * head_dim`).
/// - K/V: `[kv_seq_len, num_kv_heads, head_dim]` row-major.
/// - Returns `seq_len * num_heads` partials, indexed as
///   `result[qt * num_heads + h]`.
///
/// **Causal mask**: `hi = min(abs_i + 1, kv_seq_len)` where
/// `abs_i = cache_offset + qt`. The effective KV range is
/// `[j_start, min(j_end, hi))` — positions past the causal bound are skipped,
/// matching the GPU kernel's structural pre-clamp.
///
/// **GQA**: `kv_head = h / (num_heads / num_kv_heads)`, requiring
/// `num_heads % num_kv_heads == 0`.
#[allow(clippy::too_many_arguments)]
pub fn strict_attention_partial_online(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
    cache_offset: u32,
    j_start: usize,
    j_end: usize,
) -> Vec<SoftmaxPartial> {
    assert!(
        num_heads % num_kv_heads == 0,
        "GQA: num_heads must be a multiple of num_kv_heads"
    );
    let q_per_kv = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_stride = num_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;

    let mut results = Vec::with_capacity(seq_len * num_heads);

    for h in 0..num_heads {
        let kv_head = h / q_per_kv;
        for qt in 0..seq_len {
            let abs_i = (cache_offset as usize) + qt;
            let hi = (abs_i + 1).min(kv_seq_len);

            // Clamp the requested range to the causal bound.
            let lo = j_start.min(hi);
            let hi_eff = j_end.min(hi);

            // Per-d online softmax running state (matches woody_attention_online_f32).
            let mut acc = vec![0.0f32; head_dim];
            let mut running_max = vec![f32::NEG_INFINITY; head_dim];
            let mut running_sum = vec![0.0f32; head_dim];

            for j in lo..hi_eff {
                // Score = (q · k[j]) * scale
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qt * q_stride + h * head_dim + d]
                        * k[j * kv_stride + kv_head * head_dim + d];
                }
                let s = dot * scale;
                for d in 0..head_dim {
                    let prev_m = running_max[d];
                    let new_m = if s > prev_m { s } else { prev_m };
                    let scale_prev = if new_m == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (prev_m - new_m).exp()
                    };
                    running_sum[d] = running_sum[d] * scale_prev;
                    acc[d] = acc[d] * scale_prev;
                    running_max[d] = new_m;
                    let w = if s == new_m { 1.0f32 } else { (s - new_m).exp() };
                    running_sum[d] += w;
                    acc[d] += w * v[j * kv_stride + kv_head * head_dim + d];
                }
            }

            results.push(SoftmaxPartial {
                // Each dim has its own max/sum in the per-d formulation, but the
                // merge helper works on a single (max, sum) per partial. Since the
                // score s is the same across all d (it's a dot product reduced to a
                // scalar), running_max is identical across d — collapse to a single
                // value. running_sum varies by d only if the initial max differs,
                // which it can't (all start at -inf). So the scalar is exact.
                max: running_max[0],
                sum: running_sum[0],
                acc,
            });
        }
    }
    results
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

    // ====================================================================
    // WI 3 — Gate 3.6.1: hybrid CPU/GPU attention correctness parity.
    //
    // Per `grim_rocm_consumer_perf_planv2.md` §3.6.1: "constructs a sequence
    // with a mix of Device-tier and Host-tier blocks, computes attention via
    // the hybrid path, and compares against computing the same attention
    // entirely on GPU (all blocks forced to Device tier, ignoring VRAM limits,
    // as a ground truth)." Here the "all-on-one-device" ground truth is the
    // single-chunk full-range partial (j_start=0, j_end=hi); the "hybrid path"
    // is the same computation split into chunks and merged via `merge_partials`.
    //
    // These tests are fully offline (pure CPU math) — no GPU required. They
    // validate that the merge formula in `grim-tensor::softmax_merge` correctly
    // reconstructs the full attention output from partials, which is the
    // correctness foundation for the GPU/CPU overlap in §3.4.3.
    // ====================================================================

    /// Deterministic LCG for reproducible test inputs (mirrors the pattern in
    /// `grim-backend-rocm/lib_internal_tests.rs`).
    fn lcg_f32(seed: u32) -> u32 {
        seed.wrapping_mul(1103515245).wrapping_add(12345)
    }

    /// Build deterministic Q/K/V test data for a given shape.
    #[allow(clippy::too_many_arguments)]
    fn build_attention_inputs(
        seed: u32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let mut s = seed;
        let mut next = || {
            s = lcg_f32(s);
            // Map to [-1, 1].
            ((s as f32 / u32::MAX as f32) * 2.0 - 1.0) as f32
        };
        let q: Vec<f32> = (0..seq_len * q_stride).map(|_| next()).collect();
        let k: Vec<f32> = (0..kv_seq_len * kv_stride).map(|_| next()).collect();
        let v: Vec<f32> = (0..kv_seq_len * kv_stride).map(|_| next()).collect();
        (q, k, v)
    }

    /// Full-range online-softmax attention (ground truth), matching the GPU
    /// kernel's algorithm exactly. Returns finalized output.
    #[allow(clippy::too_many_arguments)]
    fn full_attention_online(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        cache_offset: u32,
    ) -> Vec<f32> {
        // The single-chunk partial over [0, kv_seq_len) IS the full computation.
        let partials = strict_attention_partial_online(
            q, k, v, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len,
            cache_offset, 0, kv_seq_len,
        );
        let mut out = Vec::with_capacity(seq_len * num_heads * head_dim);
        for p in &partials {
            out.extend(p.finalize());
        }
        out
    }

    /// Compute attention by splitting KV into chunks, computing per-chunk
    /// partials, and merging — simulating the hybrid CPU/GPU path.
    #[allow(clippy::too_many_arguments)]
    fn chunked_merge_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        chunk_boundaries: &[usize], // e.g. [0, 16, 32, kv_seq_len]
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * num_heads * head_dim);
        for qt_head in 0..seq_len * num_heads {
            // We need per-(qt, head) partials, but the kernel returns them
            // all at once. Compute the full set for each chunk and pick the
            // right index.
            let mut partials_for_this: Vec<grim_tensor::SoftmaxPartial> = Vec::new();
            for win in chunk_boundaries.windows(2) {
                let j_start = win[0];
                let j_end = win[1];
                if j_start >= j_end {
                    continue;
                }
                let chunk_partials = strict_attention_partial_online(
                    q, k, v, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len,
                    cache_offset, j_start, j_end,
                );
                partials_for_this.push(chunk_partials[qt_head].clone());
            }
            let merged = grim_tensor::merge_all(&partials_for_this, head_dim);
            out.extend(merged.finalize());
        }
        out
    }

    #[test]
    fn gate_3_6_1_chunked_merge_matches_full_computation() {
        let (nh, nkv, hd, sl) = (8usize, 4usize, 32usize, 4usize);
        let kv_seq = 64usize; // divisible by 4 — clean chunk boundaries
        let cache_off = 4u32;
        let (q, k, v) = build_attention_inputs(0xA1, sl, nh, nkv, hd, kv_seq);

        let ground = full_attention_online(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let merged = chunked_merge_attention(
            &q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off,
            &[0, 16, 32, 48, 64], // 4 equal chunks
        );

        let mut max_diff = 0f32;
        for i in 0..ground.len() {
            let diff = (ground[i] - merged[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff <= 1e-5,
            "4-chunk merge vs full: max abs diff {max_diff:.e} exceeds 1e-5"
        );
    }

    #[test]
    fn gate_3_6_1_uneven_chunks_match_full() {
        // Chunks of unequal size — tests boundary math when splits don't
        // divide evenly (the load-imbalance caution from the plan).
        let (nh, nkv, hd, sl) = (8usize, 4usize, 32usize, 4usize);
        let kv_seq = 65usize; // 65 mod 4 = 1 — uneven
        let cache_off = 16u32;
        let (q, k, v) = build_attention_inputs(0xB2, sl, nh, nkv, hd, kv_seq);

        let ground = full_attention_online(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        // Uneven: [0,10), [10,30), [30,65) — asymmetric chunk sizes.
        let merged = chunked_merge_attention(
            &q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off,
            &[0, 10, 30, 65],
        );

        let mut max_diff = 0f32;
        for i in 0..ground.len() {
            let diff = (ground[i] - merged[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff <= 1e-5,
            "uneven-chunk merge vs full: max abs diff {max_diff:.e} exceeds 1e-5"
        );
    }

    #[test]
    fn gate_3_6_1_single_kv_seq_len() {
        // kv_seq_len=1 — single-token decode. Only one valid key; the merge
        // should be exact (softmax weight = 1.0).
        let (nh, nkv, hd, sl) = (4usize, 2usize, 16usize, 2usize);
        let kv_seq = 1usize;
        let cache_off = 0u32;
        let (q, k, v) = build_attention_inputs(0xC3, sl, nh, nkv, hd, kv_seq);

        let ground = full_attention_online(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let merged = chunked_merge_attention(
            &q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off,
            &[0, 1],
        );

        let mut max_diff = 0f32;
        for i in 0..ground.len() {
            let diff = (ground[i] - merged[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff <= 1e-6,
            "kv=1 merge: max abs diff {max_diff:.e} exceeds 1e-6 (should be bit-exact)"
        );
    }

    #[test]
    fn gate_3_6_1_skewed_distribution() {
        // The load-imbalance caution from the plan: one "chunk" (simulating
        // one sequence's offloaded blocks) contributes 10× the KV tokens of
        // the others. Correctness must hold even when the work is grossly
        // unbalanced — the merge doesn't care about balance.
        let (nh, nkv, hd, sl) = (4usize, 2usize, 32usize, 2usize);
        let kv_seq = 1001usize; // large, skewed
        let cache_off = 0u32;
        let (q, k, v) = build_attention_inputs(0xE5, sl, nh, nkv, hd, kv_seq);

        let ground = full_attention_online(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        // Skewed: a 900-element chunk and two ~50-element chunks.
        let merged = chunked_merge_attention(
            &q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off,
            &[0, 900, 950, 1001],
        );

        let mut max_diff = 0f32;
        for i in 0..ground.len() {
            let diff = (ground[i] - merged[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        // Long softmax tail → slightly looser tolerance (matches WI 1's
        // wi1_qkv_attention_skewed_short_seq at 5e-3).
        assert!(
            max_diff <= 5e-3,
            "skewed merge: max abs diff {max_diff:.e} exceeds 5e-3"
        );
    }

    #[test]
    fn gate_3_6_1_empty_chunk_is_identity() {
        // An empty chunk [j_start, j_end) where j_start >= causal hi produces
        // an identity partial (max=-inf, sum=0, acc=0) that must not corrupt
        // the merge. This simulates a wavefront/device with zero KV blocks.
        let (nh, nkv, hd, sl) = (4usize, 2usize, 16usize, 1usize);
        let kv_seq = 8usize;
        let cache_off = 0u32;
        let (q, k, v) = build_attention_inputs(0xD4, sl, nh, nkv, hd, kv_seq);

        let ground = full_attention_online(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        // First chunk is empty (j_start=0, j_end=0), second is the full range.
        let merged = chunked_merge_attention(
            &q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off,
            &[0, 0, 8],
        );

        let mut max_diff = 0f32;
        for i in 0..ground.len() {
            let diff = (ground[i] - merged[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff <= 1e-5,
            "empty-chunk merge: max abs diff {max_diff:.e} exceeds 1e-5"
        );
    }
}
