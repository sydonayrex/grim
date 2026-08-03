//! GPU parity test for the Whisper cross-attention kernel (`grim_cross_attention`).
//!
//! Gated by `GRIM_RUN_GPU_TESTS`. Runs the same full cross-attention math
//! (softmax(Q @ K^T / sqrt(head_dim)) @ V, non-causal) on the CPU reference
//! and on the ROCm device, then compares.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Deterministic LCG for reproducible test inputs.
fn lcg_f32(seed: u32) -> u32 {
    seed.wrapping_mul(1103515245).wrapping_add(12345)
}

fn build_test_data(
    seed: u32,
    seq_len: usize,
    enc_seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut next = || {
        s = lcg_f32(s);
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|_| next())
        .collect();
    let k: Vec<f32> = (0..enc_seq * num_heads * head_dim)
        .map(|_| next())
        .collect();
    let v: Vec<f32> = (0..enc_seq * num_heads * head_dim)
        .map(|_| next())
        .collect();
    (q, k, v)
}

fn cpu_cross_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    enc_seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * d];
    for h in 0..num_heads {
        let ho = h * head_dim;
        let mut scores = vec![0.0f32; seq_len * enc_seq];
        for i in 0..seq_len {
            for j in 0..enc_seq {
                let mut sum = 0.0;
                for hk in 0..head_dim {
                    sum += q[i * d + ho + hk] * k[j * d + ho + hk];
                }
                scores[i * enc_seq + j] = sum * scale;
            }
        }
        for i in 0..seq_len {
            let mut max_v = scores[i * enc_seq];
            for j in 1..enc_seq {
                if scores[i * enc_seq + j] > max_v {
                    max_v = scores[i * enc_seq + j];
                }
            }
            let mut sum_e = 0.0;
            for j in 0..enc_seq {
                let e = (scores[i * enc_seq + j] - max_v).exp();
                scores[i * enc_seq + j] = e;
                sum_e += e;
            }
            for j in 0..enc_seq {
                scores[i * enc_seq + j] /= sum_e;
            }
        }
        for i in 0..seq_len {
            for hk in 0..head_dim {
                let mut sum = 0.0;
                for j in 0..enc_seq {
                    sum += scores[i * enc_seq + j] * v[j * d + ho + hk];
                }
                out[i * d + ho + hk] = sum;
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

#[test]
fn test_cross_attention_gpu_parity() {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        println!("[INFO] Skipped test_cross_attention_gpu_parity (requires GRIM_RUN_GPU_TESTS)");
        return;
    }

    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0) should succeed");

    let seq_len = 4_usize;
    let enc_seq = 16_usize;
    let num_heads = 4_usize;
    let head_dim = 32_usize;

    let (q_data, k_data, v_data) = build_test_data(0xBEEF, seq_len, enc_seq, num_heads, head_dim);

    let q_buf = dev
        .from_cpu(
            &q_data,
            &Shape::from_slice(&[seq_len, num_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let k_buf = dev
        .from_cpu(
            &k_data,
            &Shape::from_slice(&[enc_seq, num_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let v_buf = dev
        .from_cpu(
            &v_data,
            &Shape::from_slice(&[enc_seq, num_heads, head_dim]),
            DType::F32,
        )
        .unwrap();

    let out_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let (gpu_out, _h) = dev
        .cross_attention(
            q_buf.as_ref(),
            k_buf.as_ref(),
            v_buf.as_ref(),
            num_heads,
            head_dim,
            seq_len,
            enc_seq,
            &out_shape,
        )
        .unwrap();
    let gpu_vec = gpu_out.to_cpu_vec_f32().unwrap();

    let cpu_vec = cpu_cross_attention(
        &q_data, &k_data, &v_data, seq_len, enc_seq, num_heads, head_dim,
    );

    assert!(
        approx_close(&gpu_vec, &cpu_vec, 5e-3),
        "Cross-attention GPU output does not match CPU reference"
    );
    println!(
        "[OK] Cross-attention GPU parity verified (seq={seq_len}, enc={enc_seq}, heads={num_heads}, hd={head_dim})."
    );
}
