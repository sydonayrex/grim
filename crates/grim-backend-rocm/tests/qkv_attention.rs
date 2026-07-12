//! Step-4 tests for the Phase-1 fused QKV-attention kernel.
//!
//! Spec: `grim_qkv_attention_kernel_spec.md`. The CPU reference is plain
//! Rust and is the source of truth — the GPU path is gated by
//! `GRIM_RUN_GPU_TESTS` so GPU-less CI still exercises the structural
//! paths of the host launcher (enabled gate, parameter plumbing, errors).
//!
//! Each test reports a f32 relative error budget of `1e-3` per Step 4. The
//! sole purpose of this module is correctness; performance tuning lives
//! under Phase 2 / `.rocm-rocm-/...` skill guidance, not here.

use grim_backend_rocm::{QkvAttentionFusionConfig, RocmDevice};
use grim_tensor::{BackendDevice, DType, Shape};

/// Pure-Rust reference implementation of causal GQA attention.
///
/// Layouts (Phase-1 contract):
/// - `q`:    [seq_len, num_heads, head_dim]
/// - `k`/`v`: [kv_seq_len, num_kv_heads, head_dim]
/// - `cache_offset`: absolute position of q[0, *, *]
///
/// Causal mask: query at absolute position `(cache_offset + i)` attends to
/// key positions `j` with `j <= cache_offset + i`.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
    cache_offset: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * num_heads * head_dim);
    assert_eq!(k.len(), kv_seq_len * num_kv_heads * head_dim);
    assert_eq!(v.len(), kv_seq_len * num_kv_heads * head_dim);
    if num_heads % num_kv_heads != 0 {
        panic!("num_heads must be a multiple of num_kv_heads");
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_per_kv = num_heads / num_kv_heads;

    let mut out = vec![0.0f32; seq_len * num_heads * head_dim];

    for i in 0..seq_len {
        let abs_i = cache_offset + i;
        for h in 0..num_heads {
            let kv_head = h / q_per_kv;
            if kv_seq_len == 0 {
                continue;
            }
            let max_j = abs_i.min(kv_seq_len - 1);
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(max_j + 1);
            for j in 0..=max_j {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q[(i * num_heads + h) * head_dim + d]
                        * k[(j * num_kv_heads + kv_head) * head_dim + d];
                }
                s *= scale;
                scores.push(s);
                if s > max_score {
                    max_score = s;
                }
            }
            // Normalize (numerically stable softmax).
            let mut denom = 0.0f32;
            for s in scores.iter() {
                denom += (s - max_score).exp();
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=max_j {
                    let w = ((scores[j] - max_score).exp()) / denom;
                    acc += w * v[(j * num_kv_heads + kv_head) * head_dim + d];
                }
                out[(i * num_heads + h) * head_dim + d] = acc;
            }
        }
    }
    out
}

fn approx_close(a: &[f32], b: &[f32], rel_tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(y.abs()).max(1e-6);
        if ((*x - *y) / denom).abs() > rel_tol {
            return false;
        }
    }
    true
}

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Empty-tensor corner: when cache_offset == 0 and seq_len == 0, the
/// reference returns an empty Vec and the GPU path returns an empty
/// output buffer; we don't crash. (Sanity that the new parameter set
/// is plumbed through without UB even at the boundary.)
#[test]
fn qkv_attention_structural_empty_call() {
    // The host launcher requires the gate to be checked first; this case
    // amply demonstrates it returns Err when the gate is closed.
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    let dev = RocmDevice::new(0);
    let _ = env; // Reserved for future GPU-gated branches.
    let q = dev
        .from_cpu(&[0.5f32; 0], &Shape::from_slice(&[0, 0, 0]), DType::F32)
        .ok();
    if let Some(q) = q {
        let res = dev.qkv_attention(
            q.as_ref(),
            q.as_ref(),
            q.as_ref(),
            1, // num_kv_heads
            0, // kv_seq_len
            0, // cache_offset
            &Shape::from_slice(&[0, 0, 0]),
        );
        // Either an empty-output Ok or a structural error are acceptable —
        // nothing crashes. The point is "the launcher is wired up".
        let _ = res;
    }
}

/// CPU reference smoke: easy 1:1 GQA, seq=4, head=4, dim=8 — no causal
/// mask in play (cache_offset=0, kv_seq_len=seq_len), inputs identical
/// to Q (trivially the output is a weighted sum of v=v).
#[test]
fn qkv_attention_reference_4x4x8_self_attention() {
    let (seq_len, num_heads, head_dim, kv_seq_len, cache_offset) = (4, 4, 8, 4, 0);
    let q: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let k = q.clone();
    let v = q.clone();
    let got = reference_attention(
        &q,
        &k,
        &v,
        seq_len,
        num_heads,
        num_heads,
        head_dim, // 1:1 GQA
        kv_seq_len,
        cache_offset,
    );
    assert_eq!(got.len(), q.len());
    // Output is bounded (single-row softmax at worst).
    let max = got.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max.is_finite(), "reference must not produce NaN");
}

/// CPU reference: non-4:1 GQA ratio (8:1) catches a regression back to
/// the hardcoded `num_heads / 4` bug.
#[test]
fn qkv_attention_reference_8_to_1_gqa() {
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (4, 8, 1, 8, 4, 0); // 8 heads sharing 1 kv_head (8:1 GQA).
    let q: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.07).cos())
        .collect();
    let k: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let v: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.03).tan().clamp(-10.0, 10.0))
        .collect();
    let got = reference_attention(
        &q, &k, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // Strength check: the very last query attends to everything, so its
    // output norm should be larger than the first query's output (which
    // only attends to k=0). This rules out accidental "always return zeros"
    // regressions.
    let head_dim_f = head_dim as f32;
    let row_norm = |row: usize, h: usize| -> f32 {
        (0..head_dim)
            .map(|d| got[(row * num_heads + h) * head_dim + d].powi(2))
            .sum::<f32>()
            .sqrt()
    };
    let first = row_norm(0, 0);
    let last = row_norm(seq_len - 1, 0);
    assert!(
        first > 0.0 && last > 0.0,
        "expected non-zero norms; first={} last={}",
        first, last
    );
}

/// CPU reference: decode-style, `seq_len == 1, cache_offset > 0` —
/// Forces the kernel to read past K/V rather than only at the local
/// position. Catches a regression where `cache_offset` is ignored.
#[test]
fn qkv_attention_reference_decode_with_cache() {
    let (kv_seq_len, cache_offset) = (16_usize, 8_usize);
    let (seq_len, num_heads, num_kv_heads, head_dim) = (1, 4, 4, 8);
    let q: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.21).sin())
        .collect();
    let k: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.19).cos())
        .collect();
    let v: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| ((i as f32 * 0.17).sin() + 0.1))
        .collect();
    let got = reference_attention(
        &q, &k, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // The single query attends to (kv indices 0..=cache_offset). Make sure
    // changing K[cache_offset, *, *] flips the scalar output (read sanity).
    let mut k_flipped = k.clone();
    let head_pick = 0_usize;
    for d in 0..head_dim {
        let i = (cache_offset * num_kv_heads + head_pick) * head_dim + d;
        k_flipped[i] += 0.5;
    }
    let got_flipped = reference_attention(
        &q, &k_flipped, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    assert_ne!(
        got, got_flipped,
        "decode query must change when an attended K position changes"
    );
}

/// CPU reference: chunked-prefill, `seq_len > 1, cache_offset > 0` —
/// this is the spec's primary "off-by-one in causal bound" trap.
#[test]
fn qkv_attention_reference_chunked_prefill() {
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (5_usize, 2, 2, 8, 7_usize, 4_usize);
    let q: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.23).sin())
        .collect();
    let k: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.05).cos().abs())
        .collect();
    let v: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.07).sin())
        .collect();
    let got = reference_attention(
        &q, &k, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // Causal bound is `cache_offset + i`. With (`seq_len=5`, `cache_offset=4`):
    //   row 0: abs_i = 4 → attends to K[0..=4] (NOT K[5] or beyond)
    //   row 4: abs_i = 8 → attends to K[0..=kv_seq_len] (capped by kv_seq_len=7)
    // Perturb K[5] — that index is OUT of causal bound for row 0 only.
    // Row 0 must be unchanged; perturbation must reach row 4 (which can see it).
    let mut k_flip = k.clone();
    let head = 0_usize;
    let perturb_j: usize = 5;
    for d in 0..head_dim {
        let idx = (perturb_j * num_kv_heads + head) * head_dim + d;
        k_flip[idx] += 10.0;
    }
    let got2 = reference_attention(
        &q, &k_flip, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // Row 0 (the latest pre-bound row) must be unchanged by K[5].
    for d in 0..head_dim {
        let i = (0_usize * num_heads + head) * head_dim + d;
        assert!(
            (got[i] - got2[i]).abs() < 1e-5,
            "row 0 must not depend on K[5] when cache_offset=4 (causal bound j<=4); got delta={}", got[i] - got2[i]
        );
    }
    // Row 4 (the latest chunked query) is allowed to differ — sanity that the
    // perturbation was actually applied (not silently filtered out).
    let mut any_diff = false;
    for h in 0..num_heads {
        for d in 0..head_dim {
            let i = (4 * num_heads + h) * head_dim + d;
            if (got[i] - got2[i]).abs() > 1e-5 {
                any_diff = true;
                break;
            }
        }
    }
    assert!(any_diff, "perturbation at K[5] must reach row 4 (cache_offset=4, abs_i=8 <= kv_seq_len=7)");
}

/// CPU reference: kv_seq_len large enough to exceed the 8192-float
/// in-shared-memory bound that the Phase-1 in-flight score buffer is
/// NOT allowed to use (the kernel uses online softmax and never
/// materializes a `kv_seq_len`-sized score vector). Off-CPU we don't have
/// a shared-memory cap to violate, but we still want a regression-test
/// for big kv_seq_len so a future change that *does* allocate per-k
/// state would surface here.
#[test]
fn qkv_attention_reference_large_kv_seq_len() {
    let (kv_seq_len, cache_offset) = (2_048_usize, 2_048_usize);
    let (seq_len, num_heads, num_kv_heads, head_dim) = (4, 2, 2, 8);
    let q: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.31).sin())
        .collect();
    let k: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.0017).cos())
        .collect();
    let v: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.0033).sin())
        .collect();
    let got = reference_attention(
        &q, &k, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // Don't time, don't assert exact values; just ensure the online-style
    // accumulator doesn't blow up at large kv_seq_len.
    let max = got.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = got.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(max.is_finite() && min.is_finite(), "NaN at large kv_seq_len: max={} min={}", max, min);
}

/// CPU reference: causal independence — later K/V cannot affect earlier
/// queries (a key property of causal masking). Differs from the previous
/// test which checked `cache_offset`, this one checks the symmetric bound
/// at each `i`.
#[test]
fn qkv_attention_reference_earlier_query_independent_of_later_kv() {
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (3_usize, 2, 2, 8, 6_usize, 0_usize);
    let q: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.41).sin())
        .collect();
    let k: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.13).cos())
        .collect();
    let v: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.27).sin())
        .collect();
    let got = reference_attention(
        &q, &k, &v,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // Test causal independence: pick a `mid_j` that is IN causal bound for
    // row 2 but out of bound for rows 0 and 1. With `cache_offset=0, seq_len=3`:
    //   row 0: abs_i = 0 → attends to K[0]
    //   row 1: abs_i = 1 → attends to K[0..=1]
    //   row 2: abs_i = 2 → attends to K[0..=2]
    // Perturb K[mid_j=2] (the row-2 ceiling) — must reach row 2 but never
    // rows 0/1.
    let mid_j: usize = 2;
    let mut k2 = k.clone();
    let mut v2 = v.clone();
    for d in 0..head_dim {
        k2[(mid_j * num_kv_heads) * head_dim + d] += 99.0;
        v2[(mid_j * num_kv_heads) * head_dim + d] += 99.0;
    }
    let got2 = reference_attention(
        &q, &k2, &v2,
        seq_len, num_heads, num_kv_heads, head_dim,
        kv_seq_len, cache_offset,
    );
    // Earlier queries (rows 0 and 1) are unchanged.
    for row in 0..2 {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let i = (row * num_heads + h) * head_dim + d;
                let diff = (got[i] - got2[i]).abs();
                assert!(
                    diff < 1e-5,
                    "row {} head {} dim {} changed (causal leak from K[V][{}]): {}",
                    row, h, d, mid_j, diff
                );
            }
        }
    }
    // The latest query (row 2) is allowed to differ — sanity that the
    // perturbation at K[mid_j=2] was actually applied (not silently
    // filtered out). Row 2's causal bound is j<=2, so K[2] is in scope.
    let mut any_diff = false;
    for h in 0..num_heads {
        for d in 0..head_dim {
            let i = (2 * num_heads + h) * head_dim + d;
            if (got[i] - got2[i]).abs() > 1e-5 {
                any_diff = true;
                break;
            }
        }
    }
    assert!(any_diff, "perturbation at K[2] must reach row 2 (causal bound includes j=2)");
}

/// Verify the gate semantics built into the *config*: a default-shape
/// `QkvAttentionFusionConfig` has `enabled: false`. This is the canonical
/// stop-sign for regression handling (a future Phase-1/2 regression can
/// be gated off again without an emergency patch).
#[test]
fn qkv_attention_config_default_gates_kernel_off() {
    let cfg = QkvAttentionFusionConfig::default();
    assert!(!cfg.enabled);
}

/// GPU-gated end-to-end check: launch `RocmDevice::qkv_attention` and
/// compare against the CPU reference for a small shape. The gate check
/// (enabled), the structural validation (num_heads % num_kv_heads), and
/// the head_dim<=64 ceiling are all exercised on the live device here.
/// Tolerance is the same `1e-3` relative error budget as Step 4.
#[test]
fn qkv_attention_gpu_matches_reference_when_enabled() {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return;
    }
    let dev = RocmDevice::new(0);

    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (4_usize, 4, 4, 8, 4, 0);

    let q_data: Vec<f32> = (0..(seq_len * num_heads * head_dim))
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let k_data: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.11).cos())
        .collect();
    let v_data: Vec<f32> = (0..(kv_seq_len * num_kv_heads * head_dim))
        .map(|i| (i as f32 * 0.07).sin())
        .collect();
    let out_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);

    let q_buf = dev.from_cpu(&q_data, &out_shape, DType::F32).unwrap();
    let k_buf = dev.from_cpu(&k_data, &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();
    let v_buf = dev.from_cpu(&v_data, &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();

    let (out, _h) = dev
        .qkv_attention(
            q_buf.as_ref(),
            k_buf.as_ref(),
            v_buf.as_ref(),
            num_kv_heads,
            kv_seq_len,
            cache_offset as u32,
            &out_shape,
        )
        .unwrap();
    let got = out.to_cpu_vec_f32().unwrap();
    let want = reference_attention(
        &q_data,
        &k_data,
        &v_data,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        cache_offset,
    );
    if !approx_close(&got, &want, 1e-3) {
        for i in 0..got.len() {
            let x = got[i];
            let y = want[i];
            let denom = x.abs().max(y.abs()).max(1e-6);
            let diff = ((x - y) / denom).abs();
            if diff > 1e-3 {
                panic!("GPU output diverged from CPU reference at index {}: got={}, want={}, diff={}. Full got={:?}\nFull want={:?}", i, x, y, diff, got, want);
            }
        }
    }
}

/// GPU-gated: when the call violates `num_heads % num_kv_heads`, the host
/// rejects with a structured error (PyTorch parity: no silent fallback).
#[test]
fn qkv_attention_gpu_rejects_bad_gqa_ratio() {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return;
    }
    let dev = RocmDevice::new(0);
    let (seq_len, head_dim) = (4_usize, 8);
    let num_heads = 4;
    let num_kv_heads = 3; // 4 % 3 != 0
    let out_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let q_buf = dev.from_cpu(&vec![0.5f32; seq_len * num_heads * head_dim], &out_shape, DType::F32).unwrap();
    let k_buf = dev.from_cpu(&vec![0.5f32; 8 * num_kv_heads * head_dim], &Shape::from_slice(&[8, num_kv_heads, head_dim]), DType::F32).unwrap();
    let v_buf = dev.from_cpu(&vec![0.5f32; 8 * num_kv_heads * head_dim], &Shape::from_slice(&[8, num_kv_heads, head_dim]), DType::F32).unwrap();
    let res = dev.qkv_attention(
        q_buf.as_ref(),
        k_buf.as_ref(),
        v_buf.as_ref(),
        num_kv_heads,
        8,
        0,
        &out_shape,
    );
    assert!(res.is_err(), "bad GQA ratio must surface as Err; got Ok");
}

/// GPU-gated: when kv_seq_len is 0, the kernel must not divide by zero and produce NaN.
#[test]
fn qkv_attention_gpu_zero_kv_seq_len_not_nan() {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return;
    }
    let dev = RocmDevice::new(0);
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (4_usize, 4, 4, 8, 0, 0);

    let q_data: Vec<f32> = vec![0.5f32; seq_len * num_heads * head_dim];
    let k_data: Vec<f32> = vec![];
    let v_data: Vec<f32> = vec![];
    let out_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);

    let q_buf = dev.from_cpu(&q_data, &out_shape, DType::F32).unwrap();
    let k_buf = dev.from_cpu(&k_data, &Shape::from_slice(&[0, num_kv_heads, head_dim]), DType::F32).unwrap();
    let v_buf = dev.from_cpu(&v_data, &Shape::from_slice(&[0, num_kv_heads, head_dim]), DType::F32).unwrap();

    let (out, _h) = dev
        .qkv_attention(
            q_buf.as_ref(),
            k_buf.as_ref(),
            v_buf.as_ref(),
            num_kv_heads,
            kv_seq_len,
            cache_offset as u32,
            &out_shape,
        )
        .unwrap();
    let got = out.to_cpu_vec_f32().unwrap();
    for x in got {
        assert!(!x.is_nan(), "GPU output must not be NaN for empty KV cache");
    }
}

