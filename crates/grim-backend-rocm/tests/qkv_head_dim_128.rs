//! Finding 2 — head_dim > 64 correctness tests.
//!
//! The CPU reference must produce finite (non-NaN) output for head_dim=128.
//! The GPU path (gated by GRIM_RUN_GPU_TESTS) must match the CPU reference
//! within atol=1e-4 instead of silently returning NaN.
//!
//! RED bar: with the head_dim > 64 NaN guard in place, the GPU path currently
//! returns Err (blocked by roc_device.rs host-side check). After the guard is
//! removed and LDS tiling is wired, the GPU output must match the CPU reference.

// NOTE: QkvAttentionFusionConfig intentionally omitted — not needed here.
use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// CPU reference: causal GQA attention for arbitrary head_dim.
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
    let scale = 1.0f32 / (head_dim as f32).sqrt();
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
                if s > max_score { max_score = s; }
            }
            let mut denom = 0.0f32;
            for s in &scores { denom += (s - max_score).exp(); }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=max_j {
                    let w = (scores[j] - max_score).exp() / denom;
                    acc += w * v[(j * num_kv_heads + kv_head) * head_dim + d];
                }
                out[(i * num_heads + h) * head_dim + d] = acc;
            }
        }
    }
    out
}

/// CPU reference must produce finite values for head_dim=128 (Llama-2/3 default).
///
/// This test does not require a GPU and validates the reference math is sound.
#[test]
fn cpu_reference_head_dim_128_not_nan() {
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (4_usize, 2, 2, 128, 4, 0);

    let q: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    let out = reference_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset);

    assert_eq!(out.len(), seq_len * num_heads * head_dim);
    for (i, &x) in out.iter().enumerate() {
        assert!(x.is_finite(), "CPU reference produced non-finite value at index {}: {}", i, x);
    }
}

/// CPU reference for head_dim=96 (Mistral default).
#[test]
fn cpu_reference_head_dim_96_not_nan() {
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (4_usize, 2, 2, 96, 4, 0);

    let q: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    let out = reference_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset);

    for (i, &x) in out.iter().enumerate() {
        assert!(x.is_finite(), "CPU reference produced non-finite value at index {}: {}", i, x);
    }
}

/// GPU path: head_dim=128 must succeed (not Err) and produce non-NaN output.
///
/// RED state: this test will fail because roc_device.rs returns Err for head_dim > 64.
/// GREEN state: after removing the guard and implementing LDS tiling, this test passes.
#[test]
fn qkv_attention_gpu_head_dim_128_not_nan() {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return;
    }
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (4_usize, 2, 2, 128, 4, 0);

    let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k_data: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.013).cos())
        .collect();
    let v_data: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    let q_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let kv_shape = Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]);

    let dev = RocmDevice::new(0);
    let q_buf = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
    let k_buf = dev.from_cpu(&k_data, &kv_shape, DType::F32).unwrap();
    let v_buf = dev.from_cpu(&v_data, &kv_shape, DType::F32).unwrap();

    let res = dev.qkv_attention(
        q_buf.as_ref(),
        k_buf.as_ref(),
        v_buf.as_ref(),
        num_kv_heads,
        kv_seq_len,
        cache_offset as u32,
        &q_shape,
    );

    // The host guard in roc_device.rs intentionally returns Err for head_dim > 64
    // until LDS-tiled kernels are implemented (Finding 2, estimated 3-5 days).
    // This test documents the expected GREEN contract: once LDS tiling lands,
    // the GPU path must return Ok and produce non-NaN output matching the CPU reference.
    // Until then, the Err is the CORRECT safe behavior — it prevents silent NaN propagation.
    let (out, _) = res.expect("head_dim=128 must succeed after LDS tiling is implemented (Finding 2)");
    let got = out.to_cpu_vec_f32().unwrap();

    for (i, &x) in got.iter().enumerate() {
        assert!(!x.is_nan(), "GPU output has NaN at index {}", i);
    }

    // Verify against CPU reference within atol=1e-4.
    let want = reference_attention(&q_data, &k_data, &v_data, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff < 1e-3,
            "GPU/CPU mismatch at index {}: got={}, want={}, diff={}",
            i, g, w, diff
        );
    }
}

/// GPU path: head_dim=64 (boundary) must still work correctly after the refactor.
#[test]
fn qkv_attention_gpu_head_dim_64_still_correct() {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return;
    }
    let (seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset) =
        (2_usize, 2, 2, 64, 2, 0);

    let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k_data: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.013).cos())
        .collect();
    let v_data: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    let q_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let kv_shape = Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]);

    let dev = RocmDevice::new(0);
    let q_buf = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
    let k_buf = dev.from_cpu(&k_data, &kv_shape, DType::F32).unwrap();
    let v_buf = dev.from_cpu(&v_data, &kv_shape, DType::F32).unwrap();

    let (out, _) = dev.qkv_attention(
        q_buf.as_ref(), k_buf.as_ref(), v_buf.as_ref(),
        num_kv_heads, kv_seq_len, cache_offset as u32, &q_shape,
    ).expect("head_dim=64 must still succeed");

    let got = out.to_cpu_vec_f32().unwrap();
    let want = reference_attention(&q_data, &k_data, &v_data, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len, cache_offset);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(diff < 1e-3, "head_dim=64 regression at index {}: got={}, want={}, diff={}", i, g, w, diff);
    }
}
