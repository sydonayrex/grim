//! Online-softmax partial-merge helpers (WI 3.4.4 shared math).
//!
//! FlashAttention-style online softmax maintains a running `(max, sum, acc)`
//! triple per output dimension. When the KV loop is split across wavefronts
//! (GPU, WI 1) or across devices (CPU/GPU hybrid, WI 3), each side produces
//! an independent partial triple that must be merged into one. The merge
//! formula is numerically stable — the *result* is independent of merge order,
//! only intermediate values differ.
//!
//! This module hosts the merge in one place so the GPU intra-kernel wavefront
//! merge (`grim_qkv_attention`), the CPU partial kernel
//! (`strict_attention_partial_online`), and the cross-device CPU/GPU merge
//! (WI 3.4.4) all reference the same math. Per the plan (`grim_rocm_consumer_perf_planv2.md`
//! §3.4.4): "this is the same math in a third place now — put it in one place
//! both sides can reference."
//!
//! The formula (mirrors the GPU kernel's wave-0 merge loop at
//! `kernels/qkv_attention.rs` lines 224–236):
//! ```text
//! new_max = max(a.max, b.max)
//! scale_a = exp(a.max - new_max)
//! scale_b = exp(b.max - new_max)
//! new_sum = a.sum * scale_a + b.sum * scale_b
//! new_acc[d] = a.acc[d] * scale_a + b.acc[d] * scale_b
//! ```

/// A partial online-softmax result for one (head, query) pair.
///
/// - `max` — running maximum score seen so far (`-inf` if no keys processed).
/// - `sum` — running denominator sum (0 if no keys processed).
/// - `acc` — running weighted value accumulator, one element per head dim.
///
/// Two partials computed over disjoint KV ranges can be merged via
/// [`merge_partials`] to produce the partial that would have resulted from
/// processing both ranges in sequence.
#[derive(Debug, Clone)]
pub struct SoftmaxPartial {
    pub max: f32,
    pub sum: f32,
    pub acc: Vec<f32>,
}

impl SoftmaxPartial {
    /// Identity element for merging: merging `empty(d)` with any partial `p`
    /// returns `p` (cloned). `max = -inf` makes the `exp(-inf - new_max) = 0`
    /// scale factor annihilate the empty side's contribution.
    pub fn empty(head_dim: usize) -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
            acc: vec![0.0; head_dim],
        }
    }

    /// Finalize: divide the accumulator by the sum to produce the attention
    /// output. The zero-guard (sum ≤ 0) returns 0 for empty KV ranges — this
    /// is the "F5 guard" against NaN on sequences with no valid keys (matches
    /// the GPU kernel's `inv_sum` ternary at `qkv_attention.rs` line 238).
    pub fn finalize(&self) -> Vec<f32> {
        if self.sum <= 0.0 {
            vec![0.0; self.acc.len()]
        } else {
            let inv = 1.0 / self.sum;
            self.acc.iter().map(|&a| a * inv).collect()
        }
    }
}

/// Merge two partial online-softmax results into one.
///
/// Numerically stable: the result is invariant to the order of `a`/`b` and
/// to the grouping of a sequence of merges (associative + commutative). This
/// is the same property the GPU kernel relies on when merging 4 wavefront
/// partials in any pairwise order.
///
/// If either side is empty (`sum == 0`, `max == -inf`), the other side is
/// returned unchanged (modulo cloning) — the `exp(-inf - new_max)` scale
/// factors zero out the empty side's contributions.
pub fn merge_partials(a: &SoftmaxPartial, b: &SoftmaxPartial) -> SoftmaxPartial {
    debug_assert_eq!(
        a.acc.len(),
        b.acc.len(),
        "merge_partials: head_dim mismatch ({} != {})",
        a.acc.len(),
        b.acc.len()
    );
    let new_max = a.max.max(b.max);
    // exp(-inf - new_max) would be exp(-inf) = 0; guard the -inf case explicitly
    // to avoid NaN from inf - inf when both sides are empty.
    let scale_a = if a.max == f32::NEG_INFINITY {
        0.0
    } else {
        (a.max - new_max).exp()
    };
    let scale_b = if b.max == f32::NEG_INFINITY {
        0.0
    } else {
        (b.max - new_max).exp()
    };
    let new_sum = a.sum * scale_a + b.sum * scale_b;
    let new_acc: Vec<f32> = a
        .acc
        .iter()
        .zip(b.acc.iter())
        .map(|(&xa, &xb)| xa * scale_a + xb * scale_b)
        .collect();
    SoftmaxPartial {
        max: new_max,
        sum: new_sum,
        acc: new_acc,
    }
}

/// Merge a slice of partials left-to-right. Uses [`empty`] as the identity
/// seed so an empty slice returns zeroed partials of the given head_dim.
pub fn merge_all(partials: &[SoftmaxPartial], head_dim: usize) -> SoftmaxPartial {
    let mut acc = SoftmaxPartial::empty(head_dim);
    for p in partials {
        acc = merge_partials(&acc, p);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partial(max: f32, sum: f32, acc: &[f32]) -> SoftmaxPartial {
        SoftmaxPartial {
            max,
            sum,
            acc: acc.to_vec(),
        }
    }

    #[test]
    fn empty_is_identity_for_merge() {
        let p = partial(1.0, 2.0, &[3.0, 4.0, 5.0]);
        let e = SoftmaxPartial::empty(3);
        let merged = merge_partials(&p, &e);
        assert!((merged.max - 1.0).abs() < 1e-6);
        assert!((merged.sum - 2.0).abs() < 1e-6);
        for d in 0..3 {
            assert!((merged.acc[d] - p.acc[d]).abs() < 1e-6, "acc[{d}] mismatch");
        }
        // Symmetric: empty + p == p.
        let merged2 = merge_partials(&e, &p);
        assert!((merged2.sum - merged.sum).abs() < 1e-6);
    }

    #[test]
    fn merge_is_commutative() {
        let a = partial(0.5, 1.0, &[1.0, 2.0]);
        let b = partial(1.5, 3.0, &[0.5, 0.25]);
        let ab = merge_partials(&a, &b);
        let ba = merge_partials(&b, &a);
        assert!((ab.max - ba.max).abs() < 1e-6);
        assert!(
            (ab.sum - ba.sum).abs() < 1e-5,
            "sum differs: {} vs {}",
            ab.sum,
            ba.sum
        );
        for d in 0..2 {
            assert!((ab.acc[d] - ba.acc[d]).abs() < 1e-5, "acc[{d}] differs");
        }
    }

    #[test]
    fn merge_is_associative() {
        let a = partial(0.3, 1.0, &[2.0, 1.0]);
        let b = partial(0.7, 2.0, &[1.0, 3.0]);
        let c = partial(0.5, 1.5, &[0.5, 0.5]);
        // (a + b) + c
        let left = merge_partials(&merge_partials(&a, &b), &c);
        // a + (b + c)
        let right = merge_partials(&a, &merge_partials(&b, &c));
        assert!((left.sum - right.sum).abs() < 1e-5, "associativity: sum");
        for d in 0..2 {
            assert!(
                (left.acc[d] - right.acc[d]).abs() < 1e-5,
                "associativity: acc[{d}]"
            );
        }
    }

    #[test]
    fn finalize_divides_by_sum() {
        let p = partial(1.0, 4.0, &[8.0, 12.0]);
        let out = p.finalize();
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn finalize_zero_sum_returns_zeros() {
        let p = SoftmaxPartial::empty(4);
        let out = p.finalize();
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn merge_all_with_empty_slice_returns_empty() {
        let result = merge_all(&[], 5);
        assert_eq!(result.acc.len(), 5);
        assert_eq!(result.sum, 0.0);
        assert_eq!(result.max, f32::NEG_INFINITY);
    }

    #[test]
    fn test_merge_partials_extreme_max_delta_numerical_stability() {
        let large = partial(80.0, 1.0, &[2.0, 4.0]);
        let small = partial(0.0, 1.0, &[100.0, 200.0]);
        let merged = merge_partials(&large, &small);
        assert_eq!(merged.max, 80.0);
        let out = merged.finalize();
        assert!((out[0] - 2.0).abs() < 1e-4);
        assert!((out[1] - 4.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Mutation-resistant golden merge values (hand-derived from the formula).
    // The commutativity/associativity tests above would pass even if the
    // merge were globally scaled by a constant; these assert exact triples.
    // -----------------------------------------------------------------------

    fn close(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-5,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    #[test]
    fn merge_partials_golden_exact_max_sum_acc() {
        let a = partial(0.0, 1.0, &[1.0, 2.0]);
        let b = partial(1.0, 3.0, &[0.5, 0.25]);
        let m = merge_partials(&a, &b);
        let new_max = 1.0f32;
        let scale_a = (0.0f32 - new_max).exp();
        let scale_b = (1.0f32 - new_max).exp();
        let want_sum = 1.0 * scale_a + 3.0 * scale_b;
        let want_acc = [
            1.0 * scale_a + 0.5 * scale_b,
            2.0 * scale_a + 0.25 * scale_b,
        ];
        close(m.max, new_max, "golden max");
        close(m.sum, want_sum, "golden sum");
        close(m.acc[0], want_acc[0], "golden acc[0]");
        close(m.acc[1], want_acc[1], "golden acc[1]");
    }

    /// A scale-rescale mutant (dropping `*scale_a` on the acc term) keeps
    /// commutativity but breaks this golden value.
    #[test]
    fn merge_partials_golden_rescale_when_max_dominates() {
        let a = partial(0.0, 1.0, &[1000.0, 2000.0]);
        let b = partial(5.0, 1.0, &[1.0, 2.0]);
        let m = merge_partials(&a, &b);
        let scale_a = (0.0f32 - 5.0f32).exp();
        let scale_b = 1.0f32;
        close(m.sum, 1.0 * scale_a + 1.0 * scale_b, "rescale sum");
        close(m.acc[0], 1000.0 * scale_a + 1.0 * scale_b, "rescale acc[0]");
        close(m.acc[1], 2000.0 * scale_a + 2.0 * scale_b, "rescale acc[1]");
        assert!(
            m.acc[0] < 10.0,
            "a's large acc must be rescaled away: {}",
            m.acc[0]
        );
    }

    #[test]
    fn finalize_golden_divides_acc_by_sum() {
        let p = partial(2.0, 8.0, &[24.0, 0.0, -16.0]);
        let out = p.finalize();
        assert_eq!(out.len(), 3);
        close(out[0], 3.0, "finalize acc[0]=24/8");
        close(out[1], 0.0, "finalize acc[1]=0/8");
        close(out[2], -2.0, "finalize acc[2]=-16/8");
    }

    /// End-to-end: merge two disjoint-KV partials and finalize must equal the
    /// attention weight from recomputing softmax over the union. This is the
    /// property FlashAttention relies on for split-KV correctness.
    #[test]
    fn merge_then_finalize_matches_direct_softmax_weights() {
        let union_max = 2.0f32;
        let e = |s: f32| (s - union_max).exp();
        let denom = e(1.0) + e(2.0) + e(0.5);
        let want_attn0 = (e(1.0) * 1.0 + e(2.0) * 3.0 + e(0.5) * 5.0) / denom;
        let ea = |score: f32, local_max: f32| (score - local_max).exp();
        let a = partial(
            2.0,
            ea(1.0, 2.0) + ea(2.0, 2.0),
            &[ea(1.0, 2.0) * 1.0 + ea(2.0, 2.0) * 3.0],
        );
        let b = partial(0.5, ea(0.5, 0.5), &[ea(0.5, 0.5) * 5.0]);
        let merged = merge_partials(&a, &b);
        let out = merged.finalize();
        assert_eq!(out.len(), 1);
        close(out[0], want_attn0, "merge+finalize == direct softmax");
    }
}
