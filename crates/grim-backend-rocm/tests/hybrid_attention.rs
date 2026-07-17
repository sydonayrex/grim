//! Correctness and concurrency test for Work Item 3 (hybrid CPU/GPU attention overlap).
//!
//! Gated by `GRIM_RUN_GPU_TESTS`. Compares hybrid split attention against
//! full-GPU execution as a ground truth (Gate 3.6.1).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape, SoftmaxPartial, merge_partials};

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Deterministic LCG for reproducible test inputs.
fn lcg_f32(seed: u32) -> u32 {
    seed.wrapping_mul(1103515245).wrapping_add(12345)
}

fn build_test_data(
    seed: u32,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut next = || {
        s = lcg_f32(s);
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..seq_len * num_heads * head_dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim).map(|_| next()).collect();
    (q, k, v)
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
fn test_hybrid_cpu_gpu_attention_correctness() {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        println!("[INFO] Skipped test_hybrid_cpu_gpu_attention_correctness (requires GRIM_RUN_GPU_TESTS)");
        return;
    }

    let dev = RocmDevice::new(0);

    // Attention parameters (Llama-shaped GQA configuration)
    let seq_len = 1_usize;
    let num_heads = 8_usize;
    let num_kv_heads = 4_usize;
    let head_dim = 64_usize;
    
    // We split 48 key-value tokens (3 blocks of size 16) into:
    // - 32 device tokens (2 blocks) on GPU
    // - 16 host tokens (1 block) on CPU
    let device_seq_len = 32_usize;
    let host_seq_len = 16_usize;
    let kv_seq_len = device_seq_len + host_seq_len;
    let cache_offset = 0_usize;

    // Generate random mock KV cache blocks
    let (q_data, k_data, v_data) = build_test_data(
        0xDEAD, seq_len, num_heads, num_kv_heads, head_dim, kv_seq_len
    );

    // ────────────────────────────────────────────────────────────────────────
    // 1. Full-GPU Ground Truth
    // ────────────────────────────────────────────────────────────────────────
    let q_buf = dev.from_cpu(&q_data, &Shape::from_slice(&[seq_len, num_heads, head_dim]), DType::F32).unwrap();
    let k_buf = dev.from_cpu(&k_data, &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();
    let v_buf = dev.from_cpu(&v_data, &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();
    
    let (gpu_gt_out, _h) = dev.qkv_attention(
        q_buf.as_ref(),
        k_buf.as_ref(),
        v_buf.as_ref(),
        num_kv_heads,
        kv_seq_len,
        cache_offset as u32,
        &Shape::from_slice(&[seq_len, num_heads, head_dim]),
        None,
        None,
    ).unwrap();
    let gt_vec = gpu_gt_out.to_cpu_vec_f32().unwrap();

    // ────────────────────────────────────────────────────────────────────────
    // 2. Hybrid Execution (GPU Device blocks + CPU Host blocks)
    // ────────────────────────────────────────────────────────────────────────
    // Let device blocks reside in positions [0, 32); host blocks in [32, 48)
    let device_k = &k_data[0..device_seq_len * num_kv_heads * head_dim];
    let device_v = &v_data[0..device_seq_len * num_kv_heads * head_dim];


    // Allocate GPU buffers for device blocks
    let dev_k_buf = dev.from_cpu(device_k, &Shape::from_slice(&[device_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();
    let dev_v_buf = dev.from_cpu(device_v, &Shape::from_slice(&[device_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();

    // Allocate lightweight GPU storage to capture GPU partial metadata (max and sum)
    let meta_shape = Shape::from_slice(&[seq_len, num_heads]);
    let dev_max_buf = dev.zeros(&meta_shape, DType::F32).unwrap();
    let dev_sum_buf = dev.zeros(&meta_shape, DType::F32).unwrap();

    // Start GPU partial attention (asynchronous execution on stream 0)
    let (dev_out_buf, _handle) = dev.qkv_attention(
        q_buf.as_ref(),
        dev_k_buf.as_ref(),
        dev_v_buf.as_ref(),
        num_kv_heads,
        device_seq_len,
        cache_offset as u32,
        &Shape::from_slice(&[seq_len, num_heads, head_dim]),
        Some(dev_max_buf.as_ref()),
        Some(dev_sum_buf.as_ref()),
    ).unwrap();

    // CONCURRENT: Compute CPU-side partial attention on the host thread for host blocks
    let cpu_partials = grim_backend_cpu::strict_kernels::strict_attention_partial_online(
        &q_data,
        &k_data,
        &v_data,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        cache_offset as u32,
        device_seq_len,
        kv_seq_len,
    );

    // JOIN POINT: Synchronize and download GPU results
    let gpu_out_vec = dev_out_buf.to_cpu_vec_f32().unwrap();
    let gpu_max_vec = dev_max_buf.to_cpu_vec_f32().unwrap();
    let gpu_sum_vec = dev_sum_buf.to_cpu_vec_f32().unwrap();

    // ────────────────────────────────────────────────────────────────────────
    // 3. Merging Partials
    // ────────────────────────────────────────────────────────────────────────
    let mut hybrid_out = vec![0.0f32; seq_len * num_heads * head_dim];

    for i in 0..seq_len {
        for h in 0..num_heads {
            let meta_idx = i * num_heads + h;
            
            // Reconstruct the GPU partial state
            let max_val = gpu_max_vec[meta_idx];
            let sum_val = gpu_sum_vec[meta_idx];
            
            let mut gpu_acc = vec![0.0f32; head_dim];
            let qh_offset = meta_idx * head_dim;
            for d in 0..head_dim {
                gpu_acc[d] = gpu_out_vec[qh_offset + d] * sum_val;
            }

            let gpu_partial = SoftmaxPartial {
                max: max_val,
                sum: sum_val,
                acc: gpu_acc,
            };

            // Retrieve the corresponding CPU partial
            let cpu_partial = &cpu_partials[meta_idx];

            // Merge the two triples
            let merged = merge_partials(&gpu_partial, &cpu_partial);
            let finalized = merged.finalize();

            for d in 0..head_dim {
                hybrid_out[qh_offset + d] = finalized[d];
            }
        }
    }

    // Parity verification
    assert!(
        approx_close(&hybrid_out, &gt_vec, 1e-3),
        "Correctness Parity Violation: Hybrid CPU/GPU output does not match full GPU ground truth!"
    );
    println!("[OK]  Correctness parity verified successfully.");
}
