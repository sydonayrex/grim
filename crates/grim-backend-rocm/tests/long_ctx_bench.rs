//! Step 1: Shipped Q8_1 splitk vs F16 splitk decode microbenchmark (32k..200k, M=1 & M=2, dev0 & dev1).
//!
//! Run:
//!   cargo test --release -p grim-backend-rocm --test long_ctx_bench -- --ignored --nocapture

use grim_backend_rocm::kernels::qkv_attention::{
    launch_qkv_attention_dev_batch_q81, KvRingPolicy,
};
use grim_backend_rocm::{AttentionOps, CoreTensorOps, RocmDevice};
use grim_tensor::{DType, Shape};
use std::time::Instant;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
#[ignore]
fn bench_decode_splitk_and_prefill_wmma() {
    let num_heads = 24usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    for ord in [0usize, 1usize] {
        let dev = RocmDevice::shared(ord);
        let peak_gb_s = if ord == 0 { 640.0 } else { 320.0 };
        println!("\n================ GPU {ord} (peak {peak_gb_s:.0} GB/s) ================");
        println!("--- Shipped Decode Split-K Attention (n_heads={num_heads}, n_kv={num_kv_heads}, hd={head_dim}, SPLIT_K=8) ---");
        println!(
            "{:<6} {:<7} {:<3} {:>6} {:>9} {:>9} {:>8} {:>8}",
            "fmt", "kv_len", "M", "wgs", "med_ms", "bytes_MB", "GB/s", "%peak"
        );

        let dummy = dev.zeros(&Shape::new(vec![1]), DType::F32).expect("dummy");

        for &kv_len in &[32_768usize, 65_536, 131_072, 200_000] {
            let k_f16_a = dev
                .zeros(&Shape::new(vec![kv_len, num_kv_heads, head_dim]), DType::F16)
                .expect("k_f16_a");
            let v_f16_a = dev
                .zeros(&Shape::new(vec![kv_len, num_kv_heads, head_dim]), DType::F16)
                .expect("v_f16_a");
            let k_f16_b = dev
                .zeros(&Shape::new(vec![kv_len, num_kv_heads, head_dim]), DType::F16)
                .expect("k_f16_b");
            let v_f16_b = dev
                .zeros(&Shape::new(vec![kv_len, num_kv_heads, head_dim]), DType::F16)
                .expect("v_f16_b");

            let q81_row_bytes = num_kv_heads * (head_dim / 32) * 36;
            let k_q81_a = dev
                .zeros(&Shape::new(vec![kv_len, q81_row_bytes]), DType::U8)
                .expect("k_q81_a");
            let v_q81_a = dev
                .zeros(&Shape::new(vec![kv_len, q81_row_bytes]), DType::U8)
                .expect("v_q81_a");
            let k_q81_b = dev
                .zeros(&Shape::new(vec![kv_len, q81_row_bytes]), DType::U8)
                .expect("k_q81_b");
            let v_q81_b = dev
                .zeros(&Shape::new(vec![kv_len, q81_row_bytes]), DType::U8)
                .expect("v_q81_b");

            let total_dev = dev
                .from_cpu(&[f32::from_bits(kv_len as u32)], &Shape::new(vec![1]), DType::F32)
                .expect("total_dev");

            for &m in &[1usize, 2usize] {
                let q = dev
                    .zeros(&Shape::new(vec![m, num_heads, head_dim]), DType::F32)
                    .expect("q");
                let out_shape = Shape::new(vec![m, num_heads, head_dim]);
                let out_buf = dev.zeros(&out_shape, DType::F32).expect("out_buf");
                let out_max = dev.zeros(&Shape::new(vec![m * num_heads]), DType::F32).expect("out_max");
                let out_sum = dev.zeros(&Shape::new(vec![m * num_heads]), DType::F32).expect("out_sum");
                let co = (kv_len - m) as u32;
                let wgs = num_heads * 8 * m;

                // F16 splitk
                let f16_bytes = (kv_len * num_kv_heads * head_dim * 2 * 2) as f64;
                let _ = dev
                    .qkv_attention(
                        q.as_ref(),
                        k_f16_a.as_ref(),
                        v_f16_a.as_ref(),
                        num_kv_heads,
                        kv_len,
                        co,
                        None,
                        &out_shape,
                        None,
                        None,
                    )
                    .expect("f16 warmup");
                dev.synchronize();
                let mut times = Vec::with_capacity(7);
                for it in 0..7 {
                    let (ka, va) = if it % 2 == 0 {
                        (k_f16_a.as_ref(), v_f16_a.as_ref())
                    } else {
                        (k_f16_b.as_ref(), v_f16_b.as_ref())
                    };
                    let t0 = Instant::now();
                    let _ = dev
                        .qkv_attention(
                            q.as_ref(),
                            ka,
                            va,
                            num_kv_heads,
                            kv_len,
                            co,
                            None,
                            &out_shape,
                            None,
                            None,
                        )
                        .expect("f16 run");
                    dev.synchronize();
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                let med = median(times);
                let gbps = (f16_bytes / 1e9) / (med / 1e3);
                println!(
                    "{:<6} {:<7} {:<3} {:>6} {:>9.3} {:>9.1} {:>8.1} {:>7.1}%",
                    "F16",
                    kv_len,
                    m,
                    wgs,
                    med,
                    f16_bytes / (1024.0 * 1024.0),
                    gbps,
                    100.0 * gbps / peak_gb_s
                );

                // Q8_1 shipped DecodeGraph splitk (launch_qkv_attention_dev_batch_q81)
                let q81_bytes = (kv_len * q81_row_bytes * 2) as f64;
                let _ = launch_qkv_attention_dev_batch_q81(
                    &dev,
                    q.as_ref(),
                    k_q81_a.as_ref(),
                    v_q81_a.as_ref(),
                    out_buf.as_ref(),
                    out_max.as_ref(),
                    out_sum.as_ref(),
                    total_dev.as_ref(),
                    num_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    m as u32,
                    0,
                    inv_sqrt_d,
                    -1,
                    0.0,
                    dummy.as_ref(),
                    0,
                    0,
                    dummy.as_ref(),
                    0,
                    1,
                    0,
                    KvRingPolicy::new(0, 0),
                )
                .expect("q81 warmup");
                dev.synchronize();
                let mut times = Vec::with_capacity(7);
                for it in 0..7 {
                    let (ka, va) = if it % 2 == 0 {
                        (k_q81_a.as_ref(), v_q81_a.as_ref())
                    } else {
                        (k_q81_b.as_ref(), v_q81_b.as_ref())
                    };
                    let t0 = Instant::now();
                    let _ = launch_qkv_attention_dev_batch_q81(
                        &dev,
                        q.as_ref(),
                        ka,
                        va,
                        out_buf.as_ref(),
                        out_max.as_ref(),
                        out_sum.as_ref(),
                        total_dev.as_ref(),
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        m as u32,
                        0,
                        inv_sqrt_d,
                        -1,
                        0.0,
                        dummy.as_ref(),
                        0,
                        0,
                        dummy.as_ref(),
                        0,
                        1,
                        0,
                        KvRingPolicy::new(0, 0),
                    )
                    .expect("q81 run");
                    dev.synchronize();
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                let med = median(times);
                let gbps = (q81_bytes / 1e9) / (med / 1e3);
                println!(
                    "{:<6} {:<7} {:<3} {:>6} {:>9.3} {:>9.1} {:>8.1} {:>7.1}%",
                    "Q8_1",
                    kv_len,
                    m,
                    wgs,
                    med,
                    q81_bytes / (1024.0 * 1024.0),
                    gbps,
                    100.0 * gbps / peak_gb_s
                );
            }
        }
    }
}
