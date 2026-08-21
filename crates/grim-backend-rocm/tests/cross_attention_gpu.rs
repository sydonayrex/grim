//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test cross_attention_gpu
//! RESULT: FAILED — hipModuleLoad failed: 209. The Whisper cross-attention JIT kernel is
//!   compiled but the .hipfb binary is not registered/loaded for this test on this dual-GPU
//!   RDNA4 box. Host-side infrastructure works; the kernel is not tied to this test at runtime.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

/// Deterministic LCG for reproducible test inputs.
fn lcg_f32(seed: u32) -> u32 {
    seed.wrapping_mul(1103515245).wrapping_add(12345)
}

fn build_test_data(
    seed: u32,
    seq_len: usize,
    enc_seq: usize,
    num_heads: usize,
    num_heads_k: usize,
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
    let k: Vec<f32> = (0..enc_seq * num_heads_k * head_dim)
        .map(|_| next())
        .collect();
    let v: Vec<f32> = (0..enc_seq * num_heads_k * head_dim)
        .map(|_| next())
        .collect();
    (q, k, v)
}

/// CPU cross-attention reference with contiguous GQA grouping:
/// query head `h` attends with KV head `h / (num_heads / num_heads_k)`,
/// matching `grim_cross_attention` and `grim_qkv_attention`. [P1-13]
fn cpu_cross_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    enc_seq: usize,
    num_heads: usize,
    num_heads_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d = num_heads * head_dim;
    let dk = num_heads_k * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_per_kv = num_heads / num_heads_k;
    let mut out = vec![0.0f32; seq_len * d];
    for h in 0..num_heads {
        let kv_head = h / q_per_kv;
        let ho = h * head_dim;
        let ko = kv_head * head_dim;
        let mut scores = vec![0.0f32; seq_len * enc_seq];
        for i in 0..seq_len {
            for j in 0..enc_seq {
                let mut sum = 0.0;
                for hk in 0..head_dim {
                    sum += q[i * d + ho + hk] * k[j * dk + ko + hk];
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
                    sum += scores[i * enc_seq + j] * v[j * dk + ko + hk];
                }
                out[i * d + ho + hk] = sum;
            }
        }
    }
    out
}

/// Interleaved-GQA variant used ONLY to prove the reference above implements
/// the contiguous convention (and that the two conventions disagree for
/// num_heads_k < num_heads). Never used as the kernel contract.
fn cpu_cross_attention_interleaved(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    enc_seq: usize,
    num_heads: usize,
    num_heads_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d = num_heads * head_dim;
    let dk = num_heads_k * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * d];
    for h in 0..num_heads {
        let kv_head = h % num_heads_k;
        let ho = h * head_dim;
        let ko = kv_head * head_dim;
        let mut scores = vec![0.0f32; seq_len * enc_seq];
        for i in 0..seq_len {
            for j in 0..enc_seq {
                let mut sum = 0.0;
                for hk in 0..head_dim {
                    sum += q[i * d + ho + hk] * k[j * dk + ko + hk];
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
                    sum += scores[i * enc_seq + j] * v[j * dk + ko + hk];
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
    let env = grim_backend_rocm::gpu_test_enabled();
    if !env {
        println!("[INFO] Skipped test_cross_attention_gpu_parity (requires GRIM_GPU_TEST=1)");
        return;
    }

    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0) should succeed");

    let seq_len = 4_usize;
    let enc_seq = 16_usize;
    let num_heads = 4_usize;
    let num_heads_k = num_heads; // launch_cross_attention hardcodes nkh = num_heads
    let head_dim = 32_usize;

    let (q_data, k_data, v_data) =
        build_test_data(0xBEEF, seq_len, enc_seq, num_heads, num_heads_k, head_dim);

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
            &Shape::from_slice(&[enc_seq, num_heads_k, head_dim]),
            DType::F32,
        )
        .unwrap();
    let v_buf = dev
        .from_cpu(
            &v_data,
            &Shape::from_slice(&[enc_seq, num_heads_k, head_dim]),
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
        &q_data,
        &k_data,
        &v_data,
        seq_len,
        enc_seq,
        num_heads,
        num_heads_k,
        head_dim,
    );

    assert!(
        approx_close(&gpu_vec, &cpu_vec, 5e-3),
        "Cross-attention GPU output does not match CPU reference"
    );
    println!(
        "[OK] Cross-attention GPU parity verified (seq={seq_len}, enc={enc_seq}, heads={num_heads}, kv_heads={num_heads_k}, hd={head_dim})."
    );
}

/// P1-13: the CPU reference must implement the contiguous GQA convention the
/// kernel now uses (`kv_head = h / (num_heads/num_heads_k)`), and the two
/// conventions must genuinely disagree for num_heads_k < num_heads so a
/// regression back to interleaved grouping would be caught.
#[test]
fn cpu_reference_uses_contiguous_gqa_grouping() {
    let seq_len = 3_usize;
    let enc_seq = 8_usize;
    let num_heads = 4_usize;
    let num_heads_k = 2_usize;
    let head_dim = 8_usize;

    let (q, k, v) = build_test_data(0x6A51, seq_len, enc_seq, num_heads, num_heads_k, head_dim);

    let contiguous = cpu_cross_attention(
        &q,
        &k,
        &v,
        seq_len,
        enc_seq,
        num_heads,
        num_heads_k,
        head_dim,
    );
    let interleaved = cpu_cross_attention_interleaved(
        &q,
        &k,
        &v,
        seq_len,
        enc_seq,
        num_heads,
        num_heads_k,
        head_dim,
    );

    assert!(
        !approx_close(&contiguous, &interleaved, 1e-6),
        "GQA grouping convention is not discriminating: contiguous == interleaved for heads={num_heads}, kv_heads={num_heads_k}"
    );

    // Cross-check the contiguous reference against a direct, explicit
    // contiguous computation so the reference itself is independently valid.
    let d = num_heads * head_dim;
    let dk = num_heads_k * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_per_kv = num_heads / num_heads_k;
    let mut direct = vec![0.0f32; seq_len * d];
    for h in 0..num_heads {
        let kv_head = h / q_per_kv;
        let ho = h * head_dim;
        let ko = kv_head * head_dim;
        for i in 0..seq_len {
            let mut scores = Vec::with_capacity(enc_seq);
            for j in 0..enc_seq {
                let dot: f32 = (0..head_dim)
                    .map(|hk| q[i * d + ho + hk] * k[j * dk + ko + hk])
                    .sum();
                scores.push(dot * scale);
            }
            let max_v = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f32;
            for s in scores.iter_mut() {
                let e = (*s - max_v).exp();
                *s = e;
                sum_e += e;
            }
            for hk in 0..head_dim {
                let acc: f32 = (0..enc_seq).map(|j| scores[j] * v[j * dk + ko + hk]).sum();
                direct[i * d + ho + hk] = acc / sum_e;
            }
        }
    }
    assert!(
        approx_close(&contiguous, &direct, 1e-5),
        "CPU contiguous reference disagrees with direct contiguous computation"
    );
    println!(
        "[OK] CPU reference is contiguous-GQA (heads={num_heads}, kv_heads={num_heads_k}) and differs from interleaved."
    );
}
