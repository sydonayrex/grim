//! Step 1: Shipped Q8_1 splitk vs F16 splitk decode microbenchmark (32k..200k, M=1 & M=2, dev0 & dev1).
//!
//! Run:
//!   cargo test --release -p grim-backend-rocm --test long_ctx_bench -- --ignored --nocapture

use grim_backend_rocm::kernels::qkv_attention::{
    launch_qkv_attention_dev_batch_q81, KvRingPolicy,
};
use grim_backend_rocm::{AttentionOps, CoreTensorOps, MemoryOps, RocmDevice};
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

#[test]
#[ignore]
fn bench_prefill_wmma_v2_parity_and_tflops() {
    let num_heads = 24usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;

    // 1. Parity & bitwise determinism at 4k on GPU 0 (non-zero deterministic LCG inputs)
    {
        let dev = RocmDevice::shared(0);
        let sl = 4_096usize;
        let mut rng = 0x1234_5678_9abc_def0_u64;
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / ((1u32 << 24) as f32)) - 0.5
        };
        let q_h: Vec<f32> = (0..sl * num_heads * head_dim).map(|_| next_f32()).collect();
        let k_h: Vec<f32> = (0..sl * num_kv_heads * head_dim).map(|_| next_f32()).collect();
        let v_h: Vec<f32> = (0..sl * num_kv_heads * head_dim).map(|_| next_f32()).collect();
        let q = dev.from_cpu(&q_h, &Shape::new(vec![sl, num_heads, head_dim]), DType::F32).unwrap();
        let k = dev.from_cpu(&k_h, &Shape::new(vec![sl, num_kv_heads, head_dim]), DType::F16).unwrap();
        let v = dev.from_cpu(&v_h, &Shape::new(vec![sl, num_kv_heads, head_dim]), DType::F16).unwrap();
        let out_shape = Shape::new(vec![sl, num_heads, head_dim]);

        unsafe { std::env::set_var("GRIM_PREFILL_WMMA_V2", "0") };
        let (o1, _) = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, None, None).unwrap();
        dev.synchronize();
        let v1_vec = o1.to_cpu_vec_f32().unwrap();

        unsafe { std::env::set_var("GRIM_PREFILL_WMMA_V2", "1") };
        let (o2a, _) = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, None, None).unwrap();
        dev.synchronize();
        let v2a_vec = o2a.to_cpu_vec_f32().unwrap();

        let (o2b, _) = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, None, None).unwrap();
        dev.synchronize();
        let v2b_vec = o2b.to_cpu_vec_f32().unwrap();

        let max_abs: f32 = v1_vec.iter().zip(&v2a_vec).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let bit_eq = v2a_vec.iter().zip(&v2b_vec).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("[prefill-wmma-v2 parity @ 4k] max_abs(v1 vs v2) = {max_abs:.6e}, bitwise_deterministic = {bit_eq}");
        assert!(max_abs <= 1e-5, "v2 parity failed: max_abs={max_abs}");
        assert!(bit_eq, "v2 not bitwise deterministic");
    }

    // 2. TFLOP/s comparison (v1 vs v2) at 4k, 16k, 32k on GPU 0 and GPU 1
    for ord in [0usize, 1usize] {
        let dev = RocmDevice::shared(ord);
        println!("\n================ GPU {ord} Prefill WMMA F16 (v1 vs v2) ================");
        println!("{:<8} {:>10} {:>10} {:>10} {:>10} {:>8}", "seq_len", "v1_ms", "v2_ms", "v1_TF/s", "v2_TF/s", "speedup");
        for &sl in &[4_096usize, 16_384, 32_768] {
            let q = dev.zeros(&Shape::new(vec![sl, num_heads, head_dim]), DType::F32).unwrap();
            let k = dev.zeros(&Shape::new(vec![sl, num_kv_heads, head_dim]), DType::F16).unwrap();
            let v = dev.zeros(&Shape::new(vec![sl, num_kv_heads, head_dim]), DType::F16).unwrap();
            let out_shape = Shape::new(vec![sl, num_heads, head_dim]);
            let tflop = 2.0 * (sl as f64) * (sl as f64) * (num_heads as f64) * (head_dim as f64) / 1e12;

            let run_ver = |flag: &str| -> f64 {
                unsafe { std::env::set_var("GRIM_PREFILL_WMMA_V2", flag) };
                let _ = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, None, None).unwrap();
                dev.synchronize();
                let mut ts = Vec::with_capacity(5);
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let _ = dev.qkv_attention(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, None, None).unwrap();
                    dev.synchronize();
                    ts.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                median(ts)
            };

            let ms1 = run_ver("0");
            let ms2 = run_ver("1");
            println!(
                "{:<8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>7.2}x",
                sl, ms1, ms2, tflop / (ms1 / 1e3), tflop / (ms2 / 1e3), ms1 / ms2
            );
        }
    }
}

#[test]
#[ignore]
fn bench_prefill_wmma_q81_parity_and_tflops() {
    let num_heads = 24usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let q81_row_bytes = num_kv_heads * (head_dim / 32) * 36;

    // 1. Parity vs F16 dequantized oracle + bitwise determinism at 4k on GPU 0
    {
        let dev = RocmDevice::shared(0);
        let sl = 4_096usize;
        let mut rng = 0x55aa_1234_9abc_def0_u64;
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / ((1u32 << 24) as f32)) - 0.5
        };
        let pack_and_deq = |src: &[f32]| -> (Vec<u8>, Vec<f32>) {
            let mut bytes = Vec::with_capacity((src.len() / 32) * 36);
            let mut deq = Vec::with_capacity(src.len());
            for chunk in src.chunks_exact(32) {
                let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                let d_f16 = half::f16::from_f32(d);
                let d_eff = d_f16.to_f32();
                let inv_d = if d_eff > 0.0 { 1.0 / d_eff } else { 0.0 };
                let mut qs = [0i8; 32];
                let mut s = 0.0f32;
                for i in 0..32 {
                    let q = (chunk[i] * inv_d).round().clamp(-127.0, 127.0) as i8;
                    qs[i] = q;
                    let v = d_eff * (q as f32);
                    s += v;
                    deq.push(v);
                }
                bytes.extend_from_slice(&d_f16.to_le_bytes());
                bytes.extend_from_slice(&half::f16::from_f32(s).to_le_bytes());
                for q in qs {
                    bytes.push(q as u8);
                }
            }
            (bytes, deq)
        };

        let q_h: Vec<f32> = (0..sl * num_heads * head_dim).map(|_| next_f32()).collect();
        let k_h: Vec<f32> = (0..sl * num_kv_heads * head_dim).map(|_| next_f32()).collect();
        let v_h: Vec<f32> = (0..sl * num_kv_heads * head_dim).map(|_| next_f32()).collect();
        let (k_bytes, k_deq) = pack_and_deq(&k_h);
        let (v_bytes, v_deq) = pack_and_deq(&v_h);

        let q = dev.from_cpu(&q_h, &Shape::new(vec![sl, num_heads, head_dim]), DType::F32).unwrap();
        let k_f16 = dev.from_cpu(&k_deq, &Shape::new(vec![sl, num_kv_heads, head_dim]), DType::F16).unwrap();
        let v_f16 = dev.from_cpu(&v_deq, &Shape::new(vec![sl, num_kv_heads, head_dim]), DType::F16).unwrap();
        let k_q81 = dev.from_cpu_bytes(&k_bytes, &Shape::new(vec![sl, q81_row_bytes]), DType::U8).unwrap();
        let v_q81 = dev.from_cpu_bytes(&v_bytes, &Shape::new(vec![sl, q81_row_bytes]), DType::U8).unwrap();
        let out_shape = Shape::new(vec![sl, num_heads, head_dim]);

        let (o_ref, _) = dev.qkv_attention(q.as_ref(), k_f16.as_ref(), v_f16.as_ref(), num_kv_heads, sl, 0, None, &out_shape, None, None).unwrap();
        dev.synchronize();
        let ref_vec = o_ref.to_cpu_vec_f32().unwrap();

        let (o_a, _) = dev.qkv_attention_q81(q.as_ref(), k_q81.as_ref(), v_q81.as_ref(), num_kv_heads, sl, 0, None, &out_shape, true).unwrap();
        dev.synchronize();
        let a_vec = o_a.to_cpu_vec_f32().unwrap();

        let (o_b, _) = dev.qkv_attention_q81(q.as_ref(), k_q81.as_ref(), v_q81.as_ref(), num_kv_heads, sl, 0, None, &out_shape, true).unwrap();
        dev.synchronize();
        let b_vec = o_b.to_cpu_vec_f32().unwrap();

        let max_abs: f32 = ref_vec.iter().zip(&a_vec).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let bit_eq = a_vec.iter().zip(&b_vec).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("[prefill-wmma-q81 parity @ 4k] max_abs(f16_deq vs q81_wmma) = {max_abs:.6e}, bitwise_deterministic = {bit_eq}");
        assert!(max_abs <= 1e-4, "q81 wmma parity failed: max_abs={max_abs}");
        assert!(bit_eq, "q81 wmma not bitwise deterministic");
    }

    // 2. TFLOP/s at 4k, 16k, 32k on GPU 0 and GPU 1
    for ord in [0usize, 1usize] {
        let dev = RocmDevice::shared(ord);
        println!("\n================ GPU {ord} Prefill WMMA Q8_1 ================");
        println!("{:<8} {:>10} {:>10} {:>10}", "seq_len", "q81_ms", "TFLOP", "TFLOP/s");
        for &sl in &[4_096usize, 16_384, 32_768] {
            let q = dev.zeros(&Shape::new(vec![sl, num_heads, head_dim]), DType::F32).unwrap();
            let k = dev.zeros(&Shape::new(vec![sl, q81_row_bytes]), DType::U8).unwrap();
            let v = dev.zeros(&Shape::new(vec![sl, q81_row_bytes]), DType::U8).unwrap();
            let out_shape = Shape::new(vec![sl, num_heads, head_dim]);
            let tflop = 2.0 * (sl as f64) * (sl as f64) * (num_heads as f64) * (head_dim as f64) / 1e12;
            let _ = dev.qkv_attention_q81(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, true).unwrap();
            dev.synchronize();
            let mut ts = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = dev.qkv_attention_q81(q.as_ref(), k.as_ref(), v.as_ref(), num_kv_heads, sl, 0, None, &out_shape, true).unwrap();
                dev.synchronize();
                ts.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let med = median(ts);
            println!("{:<8} {:>10.2} {:>10.4} {:>10.2}", sl, med, tflop, tflop / (med / 1e3));
        }
    }
}
