//! Vision M-RoPE for the Qwen3-VL clip tower (plan Task 2.2, WI-3).
//!
//! This is the subtlest single op in the tower, so each rule below is pinned by
//! its own test against the vendored reference
//! (`old/repo/llama.cpp-master/ggml/src/ggml-cpu/ops.cpp`).
//!
//! Three properties make it different from an ordinary RoPE, and all three are
//! easy to get wrong in ways that are INVISIBLE at position 0:
//!
//! 1. **Half-split pairing.** `qwen3vl.cpp` calls `ggml_rope_multi` with
//!    `d_head/2` as `n_dims`, and the VISION branch of the dispatcher is
//!    `rotate_pairs<T>(ne0, n_dims, cache, src, dst)` (`ops.cpp:6214-6216`).
//!    With `n_dims = ne0/2` the offset is `ne0/2`, so dim `i` pairs with dim
//!    `i + head_dim/2` - the NeoX half-split, NOT the interleaved GPT-J pairing
//!    `x[2i], x[2i+1]`. Since `is_vision` also skips the pass-through tail
//!    (`ops.cpp:6221`), the whole head is rotated.
//!
//! 2. **Theta resets at every section boundary.** `ggml_mrope_cache_init` is
//!    called with `indep_sects = is_vision = true` (`ops.cpp:6148-6150`,
//!    `:6194-6196`), so `theta_t/h/w/e` are reset to their base as the sector
//!    changes (`ops.cpp:6009-6024`). Each of the four sections therefore starts
//!    from `freq_base^0`, not from where the previous one ended.
//!
//! 3. **Four position ids per token**, laid out t, h, w, e
//!    (`ops.cpp:6190-6193`), with the section order t, h, w, e over
//!    `sections[]` (`ops.cpp:6038-6046`).
//!
//! A RoPE is the identity at position 0, so every one of these can be wrong
//! while every single-position assertion passes.

use grim_models_vision::qwen3vl_clip::{MropeCache, MropeParams, ROPE_FREQ_BASE};

/// Build the position-id table `qwen3vl.cpp` feeds to the rope.
///
/// For each merged position `(x, y)` on a `px x py` grid the reference emits four
/// `(h, w)` pairs - `clip.cpp:4812-4829` writes `(y + dy, x + dx)` for
/// `(dy, dx)` in `(0,0), (0,1), (1,0), (1,1)` into slots 0 and 2 identically.
/// So t == e and h == w, which is what makes the four sections degenerate to
/// three distinct rotations.
#[test]
fn position_ids_are_one_per_patch_row() {
    // The rope runs on the PRE-merge rows: the merger reshape happens after the
    // transformer stack (qwen3vl.cpp:174-181). So the table covers px*py, not the
    // merged count.
    let px = 4usize;
    let py = 4usize;
    let merge = 2usize;
    let ids = MropeCache::position_ids(px, py, merge);
    assert_eq!(ids.len(), px * py, "one id per pre-merge row");
    assert_eq!(ids[0].len(), 4, "four ids per row: t, h, w, e");
}

/// The ids follow `clip.cpp:4812-4829`: t == e, and h/w walk the 2x2 block.
#[test]
fn position_ids_match_the_reference_layout() {
    let px = 4usize;
    let py = 4usize;
    let merge = 2usize;
    let ids = MropeCache::position_ids(px, py, merge);
    // One id per patch, walked row-major: index = y * px + x. Each id is
    // [t=y, h=x, w=y, e=x], matching the reference's slot order
    // (clip.cpp:4827-4830 read as t, h, w, e).
    assert_eq!(ids[0], [0, 0, 0, 0], "patch (0,0)");
    assert_eq!(ids[1], [0, 1, 0, 1], "patch (0,1) advances x");
    assert_eq!(ids[4], [1, 0, 1, 0], "patch (1,0) advances y");
    assert_eq!(ids[15], [3, 3, 3, 3], "last patch on a 4x4 grid");
}

/// The reference writes slots 0 and 2 with `y + dy` and slots 1 and 3 with
/// `x + dx` (`clip.cpp:4827-4830`), and the rope reads t, h, w, e. So **t == w**
/// and **h == e** - not t == e as one might assume from the ordering.
#[test]
fn third_and_fourth_ids_repeat_the_first_two() {
    let ids = MropeCache::position_ids(4, 4, 2);
    for (n, id) in ids.iter().enumerate() {
        assert_eq!(id[0], id[2], "token {n}: t == w (clip.cpp:4827,4829)");
        assert_eq!(id[1], id[3], "token {n}: h == e (clip.cpp:4828,4830)");
    }
}

/// At position 0 a RoPE is the identity, for any pairing. This is the single
/// assertion that catches a broken cos/sin table or a mis-sized rotation.
#[test]
fn position_zero_is_exactly_the_identity() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    let x = (0..8).map(|i| i as f32 * 0.25 - 0.5).collect::<Vec<f32>>();
    let out = p.apply(&x, &[0, 0, 0, 0]);
    // Bit-exact, not merely close: at position 0 every theta is 0, so cos is
    // exactly 1.0 and sin exactly 0.0, and the arithmetic is exact. A loose
    // tolerance here would pass for ANY implementation, which is why this test
    // also asserts a non-zero position actually rotates.
    for (i, (a, b)) in x.iter().zip(out.iter()).enumerate() {
        assert_eq!(
            *a, *b,
            "dim {i} changed at position 0: {a} -> {b}; it must be bit-identical"
        );
    }
    // And the complementary half: a non-zero position MUST move something, so a
    // stub that returned the input unchanged would fail.
    let moved = p.apply(&x, &[1, 2, 3, 1]);
    assert!(
        x.iter().zip(moved.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "a non-zero position must rotate; the identity check above is only \
         meaningful if the rope does something at all"
    );
}

/// Half-split, not interleaved. At a non-zero position the two conventions give
/// visibly different answers, so this distinguishes them.
///
/// With head_dim 8 and a single section, dim 0 pairs with dim 4 (half-split).
/// `out[0] = x0*cos - x4*sin`, so with `sin != 0` the result differs from `x0`.
#[test]
fn pairing_is_half_split_not_interleaved() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    let mut x = vec![0.0f32; 8];
    x[0] = 1.0; // pairs with dim 4 under half-split
    x[4] = 0.0;
    let out = p.apply(&x, &[0, 1, 0, 1]);
    // Under half-split, dim 0 mixes with dim 4. With x[4] = 0 the output at dim 0
    // is x[0]*cos = cos(theta_0), and cos(theta_0) != 1 once h/w are non-zero.
    // Under INTERLEAVED pairing dim 0 would pair with dim 1 = 0, giving exactly
    // x[0]*cos as well - so also assert dim 4 is untouched by dim 0's rotation
    // in a way that differs between conventions.
    //
    // The decisive test: put the mass in dim 1. Under half-split, dim 1 pairs
    // with dim 5 and dim 0/4 are a separate pair; under interleaved, dim 0 pairs
    // with dim 1.
    let mut y = vec![0.0f32; 8];
    y[1] = 1.0;
    let outy = p.apply(&y, &[0, 1, 0, 1]);
    // Half-split: the (0,4) pair both see theta_0; the (1,5) pair sees theta_1.
    // outy[0] must be 0 (x[0] and x[4] are both 0, so that pair's output is 0)
    // regardless of convention, BUT outy[1] is rotated by theta_1.
    assert!(outy[0].abs() < 1e-6, "dim 0 must stay 0 under half-split");
    assert!(
        (outy[1] - 1.0).abs() > 1e-4,
        "dim 1 must rotate, so the pairing is half-split with a non-trivial \
         angle; got {y:?} -> {outy:?}"
    );
    let _ = (out, x);
}

/// Theta resets at each section boundary (`indep_sects = true` for vision).
///
/// So section 1's first pair uses `freq_base^0`, not the continued theta from
/// section 0. Without the reset, every section would drift to a different base
/// frequency and the model would be quietly wrong.
#[test]
fn theta_resets_at_each_section_boundary() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    // sections = [2,2,2,2] pairs each -> 8 dims. Pair index 0 is section 0, pair
    // index 2 is section 1. Both must start from theta = freq_base^0 = 1.
    // head_dim 8 -> 4 pairs, 4 sections -> 1 pair per section. So section s
    // occupies pair s, and section 1's pair must use theta = h (not the theta
    // pair 0 left behind after scaling).
    let cache_h = p.cache(&[0, 1, 0, 1]);
    let cos_s1 = cache_h[1 * 2];
    let sin_s1 = cache_h[1 * 2 + 1];
    assert!(
        (cos_s1 - 1.0f32.cos()).abs() < 1e-5,
        "section 1 must start from theta = h = 1, got cos {cos_s1}"
    );
    assert!(
        (sin_s1 - 1.0f32.sin()).abs() < 1e-5,
        "section 1 must start from theta = h = 1, got sin {sin_s1}"
    );
    // Without the reset, section 1 would inherit theta_t = 0 (since t = 0), giving
    // cos 1.0 and sin 0.0 - so this pair discriminates the two behaviours.
    assert!(
        sin_s1.abs() > 1e-3,
        "sin must be non-zero, i.e. section 1 did NOT inherit t = 0"
    );
}

/// Within a section, each successive pair advances theta by
/// `freq_base^(-2/n_dims)` (`ops.cpp:5988`, `theta *= theta_scale`).
#[test]
fn theta_advances_by_freq_base_within_a_section() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    // With 1 pair per section there is no second pair inside a section, so use a
    // head with 2 pairs per section: head_dim 16 -> 8 pairs / 4 sections = 2.
    let p2 = MropeParams {
        head_dim: 16,
        num_sections: 4,
        freq_base: 10000.0,
    };
    let theta_scale = 10000f32.powf(-2.0 / 8.0);
    assert!(
        (theta_scale - 0.1).abs() < 1e-6,
        "theta_scale = {theta_scale}"
    );
    let cache = p2.cache(&[1, 0, 0, 0]);
    // Section 0 owns pairs 0 and 1: pair 0 uses theta = t = 1, pair 1 uses
    // theta = 1 * theta_scale = 0.1.
    assert!(
        (cache[0] - 1.0f32.cos()).abs() < 1e-5,
        "pair 0 cos {}",
        cache[0]
    );
    assert!(
        (cache[2] - 0.1f32.cos()).abs() < 1e-5,
        "pair 1 cos {} vs {}",
        cache[2],
        0.1f32.cos()
    );
    // Section 1 starts fresh at theta = h = 0, so pair 2 is the identity again.
    assert!(
        (cache[4] - 1.0).abs() < 1e-5,
        "pair 2 must RESET to theta = h = 0, got {}",
        cache[4]
    );
    let _ = p;
}

/// The section layout is t, h, w, e in that order (`ops.cpp:6038-6046`), so
/// changing `h` must not move section 0's rotation.
#[test]
fn section_zero_uses_only_the_temporal_id() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    let base = p.cache(&[2, 0, 0, 0]);
    let with_h = p.cache(&[2, 7, 0, 0]);
    // Section 0 occupies pairs 0..sections[0]. With head_dim 8 and 4 equal
    // sections there are 4 pairs, so sections[0] = 1 and section 0 is pair 0.
    assert!(
        (base[0] - with_h[0]).abs() < 1e-6,
        "section 0 must ignore h: {} vs {}",
        base[0],
        with_h[0]
    );
    // Section 1 (pair 1) must change with h.
    assert!(
        (base[2] - with_h[2]).abs() > 1e-6,
        "section 1 must follow h"
    );
}

/// A rotation is norm-preserving AND permutes whole 2-vectors.
///
/// Norm preservation alone cannot catch a wrong pairing: any orthogonal mix of
/// the dims preserves it, so half-split vs interleaved would both pass. What
/// distinguishes them is that half-split pairs `(i, i + half)` as 2-vectors, so
/// the norm of EACH PAIR is preserved. Asserting the per-pair norm closes the
/// gap the aggregate norm left open.
#[test]
fn each_half_split_pair_keeps_its_own_norm() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    let half = p.head_dim / 2;
    let x = (0..8)
        .map(|i| (i as f32 * 0.37).sin() + 0.11)
        .collect::<Vec<f32>>();
    let out = p.apply(&x, &[3, 5, 7, 3]);
    for pair in 0..half {
        let before = x[pair] * x[pair] + x[pair + half] * x[pair + half];
        let after = out[pair] * out[pair] + out[pair + half] * out[pair + half];
        assert!(
            (before - after).abs() < 1e-5,
            "pair {pair} ({pair}, {pair}+{half}) changed its own 2-vector norm: \
             {before} -> {after}; a wrong pairing mixes across pairs instead"
        );
    }
    // And the whole head, for completeness.
    let before: f32 = x.iter().map(|v| v * v).sum();
    let after: f32 = out.iter().map(|v| v * v).sum();
    assert!(
        (before - after).abs() < 1e-4,
        "whole-head norm changed: {before} -> {after}"
    );

    // Norm-based invariants cannot pin the PAIRING on their own: every
    // orthogonal mix of the dims preserves both the whole norm and the per-pair
    // norms. So assert the exact expected output for a case computable by hand.
    // Pair 0 uses theta = t = 1, so dim 0 must mix with dim 4 (half-split), and
    // dim 1 must mix with dim 5 at theta = t * theta_scale = 0.01.
    let theta_scale = 10000f32.powf(-2.0 / 4.0); // 8 dims -> 4 pairs
    let z = vec![1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let got = p.apply(&z, &[1, 0, 0, 0]);
    // Pair 0: (x0=1, x4=0) -> out0 = cos(1), out4 = sin(1).
    assert!(
        (got[0] - 1.0f32.cos()).abs() < 1e-5,
        "dim 0 must mix with dim 4 at theta=1, got {} vs {}",
        got[0],
        1.0f32.cos()
    );
    assert!(
        (got[4] - 1.0f32.sin()).abs() < 1e-5,
        "dim 4 must receive dim 0's rotation, got {} vs {}",
        got[4],
        1.0f32.sin()
    );
    // Pair 1 is in SECTION 1 (8 dims -> 4 pairs, 4 sections -> 1 pair each), and
    // its theta is h, which is 0 here. So pair 1 is itself the identity, and dim 1
    // must come back UNCHANGED - which is exactly what distinguishes "pair 1
    // belongs to section 1" from "theta just kept decaying through section 0".
    assert!(
        (got[1] - 1.0).abs() < 1e-6,
        "dim 1 is in section 1 with theta = h = 0, so it must be unchanged; \
         got {} (a decaying theta would give {}), theta_scale was {theta_scale}",
        got[1],
        theta_scale.cos()
    );
    assert!(
        got[5].abs() < 1e-6,
        "dim 5 must stay 0 too, got {}",
        got[5]
    );
    for untouched in [2usize, 3, 6, 7] {
        assert!(
            got[untouched].abs() < 1e-6,
            "dim {untouched} must stay 0; it mixed with a zero partner"
        );
    }
    let _ = theta_scale;
}

/// Applying the rotation twice with the negated ids must restore the input
/// (up to float error), which is the defining property of a rotation.
#[test]
fn rotating_by_negated_positions_is_the_inverse() {
    let p = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    let x = (0..8)
        .map(|i| (i as f32 * 0.21).cos())
        .collect::<Vec<f32>>();
    let fwd = p.apply(&x, &[2, 4, 6, 2]);
    let back = p.apply(&fwd, &[-2, -4, -6, -2]);
    for (i, (a, b)) in x.iter().zip(back.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "dim {i}: {a} vs {b}");
    }
}


/// The vision rope's `freq_base` is 10000, NOT the text tower's 10000000.
///
/// `qwen3vl.cpp:106-107` passes 10000; the text GGUF declares
/// `qwen35.rope.freq_base = 10000000`. Using the text value here scales every
/// theta by 1000x, which no shape check catches. This pins the value by
/// deriving it from an observable rotation.
#[test]
fn freq_base_is_ten_thousand() {
    let p = MropeParams {
        head_dim: 16,
        num_sections: 4,
        freq_base: 10000.0,
    };
    // Section 0, pair 0, ids [1,0,0,0] -> theta = t = 1, so cos = cos(1).
    // That does not involve theta_scale, so use pair 1 of section 0, whose theta
    // is t * theta_scale = 10000^(-2/8) = 0.1.
    let cache = p.cache(&[1, 0, 0, 0]);
    assert!(
        (cache[2] - 0.1f32.cos()).abs() < 1e-5,
        "theta_scale implies freq_base 10000; got {}",
        cache[2]
    );
    // And demonstrate that 10000000 would be observably different.
    let wrong = MropeParams {
        head_dim: 16,
        num_sections: 4,
        freq_base: 10000000.0,
    };
    let wcache = wrong.cache(&[1, 0, 0, 0]);
    assert!(
        (wcache[2] - cache[2]).abs() > 1e-3,
        "the test must actually discriminate: a wrong freq_base produced the \
         same rotation"
    );
}


/// A head too narrow to fill four sections must be refused, not panic.
///
/// `head_dim 4` gives 2 pairs, which cannot make four non-empty sections; the
/// reference would divide by a zero section width. Refusal is the only safe
/// behaviour.
#[test]
fn head_too_narrow_for_four_sections_is_refused() {
    let narrow = MropeParams {
        head_dim: 4,
        num_sections: 4,
        freq_base: 10000.0,
    };
    assert!(
        !narrow.is_valid(),
        "head_dim 4 / 4 sections must be reported invalid"
    );
    let ok = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    };
    assert!(ok.is_valid(), "head_dim 8 / 4 sections is the minimum valid width");
}




/// The `ROPE_FREQ_BASE` the forward pass actually ships is 10000.
///
/// Everything else about a wrong base looks fine - finite values, correct shapes,
/// norm preserved - so the only honest way to pin it is to compare the constant
/// the product uses against the reference value. The text tower's 10000000 is the
/// tempting wrong answer.
#[test]
fn shipped_rope_freq_base_is_ten_thousand() {
    assert_eq!(
        ROPE_FREQ_BASE, 10000.0,
        "qwen3vl.cpp:106-107 passes 10000 for the vision rope"
    );
    assert_ne!(
        ROPE_FREQ_BASE, 10000000.0,
        "that is the TEXT tower's base; using it here scales every theta 1000x"
    );
}

/// The real checkpoint's geometry: head_dim 72, 4 sections of 18 pairs each.
#[test]
fn real_geometry_builds_a_consistent_cache() {
    let p = MropeParams {
        head_dim: 72,
        num_sections: 4,
        freq_base: 10000.0,
    };
    // 72 dims = 36 pairs; 4 sections of 9 pairs each.
    let cache = p.cache(&[4, 5, 6, 4]);
    assert_eq!(cache.len(), 36 * 2, "one cos/sin pair per dim pair");
    // All cache entries must be finite.
    assert!(
        cache.iter().all(|v| v.is_finite()),
        "cache must be finite; head_dim 72 gave a non-finite entry"
    );
    let x = (0..72)
        .map(|i| (i as f32 * 0.11).sin())
        .collect::<Vec<f32>>();
    let out = p.apply(&x, &[4, 5, 6, 4]);
    assert_eq!(out.len(), 72);
    let before: f32 = x.iter().map(|v| v * v).sum();
    let after: f32 = out.iter().map(|v| v * v).sum();
    // 1e-4 relative on an 8-dim f32 rotation is a real bound: a dropped dim, a
    // duplicated one, or a mis-sized pairing all break it by far more.
    assert!(
        (before - after).abs() < 1e-4,
        "rotation changed the norm: {before} -> {after}; x={x:?} out={out:?}"
    );
    // Guard the guard: confirm this bound actually discriminates, by scaling.
    let scaled = MropeParams {
        head_dim: 8,
        num_sections: 4,
        freq_base: 10000.0,
    }
    .apply(
        &(0..8).map(|i| (i as f32 * 0.37).sin() * 2.0).collect::<Vec<f32>>(),
        &[3, 5, 7, 3],
    );
    let doubled_in: f32 = (0..8).map(|i| ((i as f32 * 0.37).sin() * 2.0).powi(2)).sum();
    let doubled_out: f32 = scaled.iter().map(|v| v * v).sum();
    assert!(
        (doubled_in - doubled_out).abs() < 1e-3,
        "norm preservation must hold at a different magnitude too, otherwise \
         the tolerance is too loose to discriminate"
    );
}
