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

#[test]
#[ignore]
fn test_gdn_v3_reg128_vs_v2_vs_fp64() {
    use grim_backend_rocm::RecurrentOps;
    let dev = RocmDevice::shared(0);
    let num_v = 4usize;
    let num_k = 2usize;
    let head_dim = 128usize;
    let key_dim = num_k * head_dim;
    let value_dim = num_v * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;
    let eps64 = 1e-6f64;

    println!("\n=== GDN v2 vs v3_reg128 vs FP64 CPU reference (high-memory decay ~0.99) ===");
    println!("Pass criterion: v3_rel_L2 <= max(1e-6, 2.0 * v2_rel_L2) at EVERY length");
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "seq_len", "v2_out_L2", "v3_out_L2", "v2_st_L2", "v3_st_L2", "v2_vs_v3_max"
    );

    for &seq_len in &[2_048usize, 32_768, 131_072, 200_000, 262_144] {
        let mut rng = 0xdead_beef_cafe_1234_u64 ^ (seq_len as u64);
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / ((1u32 << 24) as f32)) - 0.5
        };

        let conv_h: Vec<f32> = (0..seq_len * conv_dim).map(|_| next_f32() * 0.5).collect();
        // Realistic long-memory gates: alpha + dt_bias ~ -4.5, ssm_a ~ -1.0 -> softplus ~ 0.01 -> decay ~ 0.99
        let alpha_h: Vec<f32> = (0..seq_len * num_v).map(|_| -4.5 + next_f32() * 0.5).collect();
        let beta_h: Vec<f32> = (0..seq_len * num_v).map(|_| -1.0 + next_f32()).collect();
        let dt_bias_h = vec![0.0f32; num_v];
        let ssm_a_h = vec![-1.0f32; num_v];
        let norm_w_h = vec![1.0f32; head_dim];

        // FP64 CPU reference scan
        let mut s64 = vec![0.0f64; num_v * head_dim * head_dim];
        let mut out64 = vec![0.0f64; seq_len * value_dim];
        let rstd64 = 1.0f64 / (head_dim as f64).sqrt();
        let silu64 = |x: f64| x / (1.0 + (-x).exp());
        let softplus64 = |x: f64| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

        for t in 0..seq_len {
            let ct = &conv_h[t * conv_dim..(t + 1) * conv_dim];
            for h in 0..num_v {
                let kh = h % num_k;
                let mut nk = [0.0f64; 128];
                let mut nq = [0.0f64; 128];
                let mut vv = [0.0f64; 128];
                let mut k2 = 0.0f64;
                let mut q2 = 0.0f64;
                for j in 0..head_dim {
                    let qq = silu64(ct[kh * head_dim + j] as f64);
                    let kk = silu64(ct[key_dim + kh * head_dim + j] as f64);
                    nq[j] = qq;
                    nk[j] = kk;
                    vv[j] = silu64(ct[2 * key_dim + h * head_dim + j] as f64);
                    k2 += kk * kk;
                    q2 += qq * qq;
                }
                let kden = (k2 + eps64).sqrt();
                let qden = (q2 + eps64).sqrt();
                for j in 0..head_dim {
                    nk[j] /= kden;
                    nq[j] /= qden;
                }
                let gate = softplus64(alpha_h[t * num_v + h] as f64 + dt_bias_h[h] as f64) * (ssm_a_h[h] as f64);
                let beta = 1.0 / (1.0 + (-(beta_h[t * num_v + h] as f64)).exp());
                let decay = gate.exp();

                let mut a_vec = [0.0f64; 128];
                let mut ss = 0.0f64;
                for j in 0..head_dim {
                    let row_off = (h * head_dim + j) * head_dim;
                    let mut pred = 0.0f64;
                    for i in 0..head_dim {
                        pred += nk[i] * (decay * s64[row_off + i]);
                    }
                    let delta = beta * (vv[j] - pred);
                    let mut a = 0.0f64;
                    for i in 0..head_dim {
                        let s2 = decay * s64[row_off + i] + nk[i] * delta;
                        s64[row_off + i] = s2;
                        a += nq[i] * s2;
                    }
                    let a_sc = a * rstd64;
                    a_vec[j] = a_sc;
                    ss += a_sc * a_sc;
                }
                let inv_rms = 1.0 / (ss * rstd64 * rstd64 + eps64).sqrt();
                for j in 0..head_dim {
                    out64[t * value_dim + h * head_dim + j] = a_vec[j] * inv_rms * (norm_w_h[j] as f64);
                }
            }
        }

        let conv_d = dev.from_cpu(&conv_h, &Shape::new(vec![seq_len, conv_dim]), DType::F32).unwrap();
        let alpha_d = dev.from_cpu(&alpha_h, &Shape::new(vec![seq_len, num_v]), DType::F32).unwrap();
        let beta_d = dev.from_cpu(&beta_h, &Shape::new(vec![seq_len, num_v]), DType::F32).unwrap();
        let dt_d = dev.from_cpu(&dt_bias_h, &Shape::new(vec![num_v]), DType::F32).unwrap();
        let a_d = dev.from_cpu(&ssm_a_h, &Shape::new(vec![num_v]), DType::F32).unwrap();
        let nw_d = dev.from_cpu(&norm_w_h, &Shape::new(vec![head_dim]), DType::F32).unwrap();

        let run_gpu = |flag: &str| -> (Vec<f32>, Vec<f32>) {
            unsafe { std::env::set_var("GRIM_KDA_SCAN_REG", flag) };
            let s_d = dev.zeros(&Shape::new(vec![num_v, head_dim, head_dim]), DType::F32).unwrap();
            let (o_d, _) = dev
                .kda_gated_delta_rule_scan(
                    conv_d.as_ref(),
                    alpha_d.as_ref(),
                    beta_d.as_ref(),
                    dt_d.as_ref(),
                    a_d.as_ref(),
                    nw_d.as_ref(),
                    None,
                    s_d.as_ref(),
                    seq_len,
                    num_v,
                    num_k,
                    head_dim,
                    eps,
                    &Shape::new(vec![seq_len, value_dim]),
                )
                .unwrap();
            dev.synchronize();
            (o_d.to_cpu_vec_f32().unwrap(), s_d.to_cpu_vec_f32().unwrap())
        };

        let (o2, s2) = run_gpu("0");
        let (o3, s3) = run_gpu("1");

        let rel_l2 = |got: &[f32], ref64: &[f64]| -> f64 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (&g, &r) in got.iter().zip(ref64) {
                let d = (g as f64) - r;
                num += d * d;
                den += r * r;
            }
            (num / den.max(1e-30)).sqrt()
        };
        let v2_o = rel_l2(&o2, &out64);
        let v3_o = rel_l2(&o3, &out64);
        let v2_s = rel_l2(&s2, &s64);
        let v3_s = rel_l2(&s3, &s64);
        let v2_vs_v3: f32 = o2.iter().zip(&o3).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!(
            "{:<8} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}",
            seq_len, v2_o, v3_o, v2_s, v3_s, v2_vs_v3
        );
        assert!(v3_o <= (2.0 * v2_o).max(1e-6), "v3 out L2 exceeded 2x v2 at {seq_len}");
        assert!(v3_s <= (2.0 * v2_s).max(1e-6), "v3 state L2 exceeded 2x v2 at {seq_len}");
    }
}

/// Phase 0A TDD gate: eager `qkv_attention_q81` (seq_len <= 2) must match the shipped
/// graph `launch_qkv_attention_dev_batch_q81` (GQA SPLIT_K=64) bitwise at 2k and stay
/// within 2.0x of graph latency at 32k.
#[test]
#[ignore]
fn eager_q81_matches_graph_decode() {
    let dev = RocmDevice::shared(0);
    let num_heads = 24usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
    let q81_row_bytes = num_kv_heads * (head_dim / 32) * 36;
    let dummy = dev.zeros(&Shape::new(vec![4]), DType::F32).unwrap();

    // 1. Bitwise parity at 2k for seq_len = 1 and 2
    let kv_len = 2_048usize;
    let mut rng = 0x1122_3344_5566_7788_u64;
    let mut next_f32 = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 40) as f32 / ((1u32 << 24) as f32)) - 0.5
    };
    let pack_q81 = |src: &[f32]| -> Vec<u8> {
        let mut out = Vec::with_capacity((src.len() / 32) * 36);
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
                s += d_eff * (q as f32);
            }
            out.extend_from_slice(&d_f16.to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(s).to_le_bytes());
            for q in qs {
                out.push(q as u8);
            }
        }
        out
    };
    let k_h: Vec<f32> = (0..kv_len * num_kv_heads * head_dim).map(|_| next_f32()).collect();
    let v_h: Vec<f32> = (0..kv_len * num_kv_heads * head_dim).map(|_| next_f32()).collect();
    let k_d = dev.from_cpu_bytes(&pack_q81(&k_h), &Shape::new(vec![kv_len, q81_row_bytes]), DType::U8).unwrap();
    let v_d = dev.from_cpu_bytes(&pack_q81(&v_h), &Shape::new(vec![kv_len, q81_row_bytes]), DType::U8).unwrap();
    let tot_d = dev.from_cpu(&[f32::from_bits(kv_len as u32)], &Shape::new(vec![1]), DType::F32).unwrap();

    for m in [1usize, 2usize] {
        let q_h: Vec<f32> = (0..m * num_heads * head_dim).map(|_| next_f32() * 4.0).collect();
        let q_d = dev.from_cpu(&q_h, &Shape::new(vec![m, num_heads, head_dim]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![m, num_heads, head_dim]);
        let g_out = dev.zeros(&out_shape, DType::F32).unwrap();
        let om = dev.zeros(&Shape::new(vec![m * num_heads]), DType::F32).unwrap();
        let os = dev.zeros(&Shape::new(vec![m * num_heads]), DType::F32).unwrap();

        launch_qkv_attention_dev_batch_q81(
            &dev, q_d.as_ref(), k_d.as_ref(), v_d.as_ref(), g_out.as_ref(),
            om.as_ref(), os.as_ref(), tot_d.as_ref(),
            num_heads as u32, num_kv_heads as u32, head_dim as u32, m as u32, 0,
            inv_sqrt_d, -1, 0.0, dummy.as_ref(), 0, 0, dummy.as_ref(), 0, 1, 0,
            KvRingPolicy::new(0, 0),
        ).unwrap();
        dev.synchronize();
        let graph_vec = g_out.to_cpu_vec_f32().unwrap();

        let (e_out, _) = dev.qkv_attention_q81(
            q_d.as_ref(), k_d.as_ref(), v_d.as_ref(),
            num_kv_heads, kv_len, (kv_len - m) as u32, None, &out_shape, true,
        ).unwrap();
        dev.synchronize();
        let eager_vec = e_out.to_cpu_vec_f32().unwrap();

        let bit_eq = graph_vec.iter().zip(&eager_vec).all(|(a, b)| a.to_bits() == b.to_bits());
        let max_abs: f32 = graph_vec.iter().zip(&eager_vec).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("[0A parity @ 2k M={m}] max_abs={max_abs:.6e}, bit_eq={bit_eq}");
        assert!(bit_eq, "eager_q81_matches_graph_decode failed bitwise parity at M={m}: max_abs={max_abs}");
    }

    // 2. Perf gate at 32k M=1: eager must be within 2.0x of graph GQA split-K
    let kv32 = 32_768usize;
    let k32 = dev.zeros(&Shape::new(vec![kv32, q81_row_bytes]), DType::U8).unwrap();
    let v32 = dev.zeros(&Shape::new(vec![kv32, q81_row_bytes]), DType::U8).unwrap();
    let tot32 = dev.from_cpu(&[f32::from_bits(kv32 as u32)], &Shape::new(vec![1]), DType::F32).unwrap();
    let q1 = dev.zeros(&Shape::new(vec![1, num_heads, head_dim]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![1, num_heads, head_dim]);
    let g_out = dev.zeros(&out_shape, DType::F32).unwrap();
    let om = dev.zeros(&Shape::new(vec![num_heads]), DType::F32).unwrap();
    let os = dev.zeros(&Shape::new(vec![num_heads]), DType::F32).unwrap();

    let mut gt = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = Instant::now();
        launch_qkv_attention_dev_batch_q81(
            &dev, q1.as_ref(), k32.as_ref(), v32.as_ref(), g_out.as_ref(),
            om.as_ref(), os.as_ref(), tot32.as_ref(),
            num_heads as u32, num_kv_heads as u32, head_dim as u32, 1, 0,
            inv_sqrt_d, -1, 0.0, dummy.as_ref(), 0, 0, dummy.as_ref(), 0, 1, 0,
            KvRingPolicy::new(0, 0),
        ).unwrap();
        dev.synchronize();
        gt.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut et = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = dev.qkv_attention_q81(
            q1.as_ref(), k32.as_ref(), v32.as_ref(),
            num_kv_heads, kv32, (kv32 - 1) as u32, None, &out_shape, true,
        ).unwrap();
        dev.synchronize();
        et.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let g_ms = median(gt);
    let e_ms = median(et);
    println!("[0A perf @ 32k M=1] graph={g_ms:.3} ms, eager={e_ms:.3} ms (ratio={:.2}x)", e_ms / g_ms);
    assert!(e_ms <= 2.0 * g_ms, "eager Q8_1 too slow: {e_ms:.3} ms vs graph {g_ms:.3} ms");
}

#[test]
#[ignore]
fn test_gdn_v4_wy32_vs_v3_and_speed() {
    use grim_backend_rocm::RecurrentOps;
    let dev = RocmDevice::new(0);
    let num_v = 48usize;
    let num_k = 16usize;
    let hd = 128usize;
    let key_dim = num_k * hd;
    let val_dim = num_v * hd;
    let conv_dim = 2 * key_dim + val_dim;
    let eps = 1e-6f32;

    for &seq_len in &[2048usize, 4096] {
        let mut rng = 0x1234_5678_9abc_def0u64 ^ (seq_len as u64);
        let mut next_f = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let conv_h: Vec<f32> = (0..seq_len * conv_dim).map(|_| next_f() * 0.5).collect();
        let alpha_h: Vec<f32> = (0..seq_len * num_v).map(|_| -2.0 + next_f() * 0.5).collect();
        let beta_h: Vec<f32> = (0..seq_len * num_v).map(|_| next_f() * 0.5).collect();
        let dt_b_h: Vec<f32> = (0..num_v).map(|_| -1.0).collect();
        let ssm_a_h: Vec<f32> = (0..num_v).map(|_| -0.1).collect();
        let nw_h: Vec<f32> = vec![1.0f32; hd];

        let conv_d = dev.from_cpu(&conv_h, &Shape::new(vec![seq_len, conv_dim]), DType::F32).unwrap();
        let alpha_d = dev.from_cpu(&alpha_h, &Shape::new(vec![seq_len, num_v]), DType::F32).unwrap();
        let beta_d = dev.from_cpu(&beta_h, &Shape::new(vec![seq_len, num_v]), DType::F32).unwrap();
        let dt_b_d = dev.from_cpu(&dt_b_h, &Shape::new(vec![num_v]), DType::F32).unwrap();
        let ssm_a_d = dev.from_cpu(&ssm_a_h, &Shape::new(vec![num_v]), DType::F32).unwrap();
        let nw_d = dev.from_cpu(&nw_h, &Shape::new(vec![hd]), DType::F32).unwrap();

        let s_v3 = dev.zeros(&Shape::new(vec![num_v, hd, hd]), DType::F32).unwrap();
        let s_v4 = dev.zeros(&Shape::new(vec![num_v, hd, hd]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![seq_len, val_dim]);

        unsafe { std::env::set_var("GRIM_KDA_SCAN_CHUNKED", "0"); }
        let (out_v3, _) = dev.kda_gated_delta_rule_scan(
            conv_d.as_ref(), alpha_d.as_ref(), beta_d.as_ref(),
            dt_b_d.as_ref(), ssm_a_d.as_ref(), nw_d.as_ref(), None,
            s_v3.as_ref(), seq_len, num_v, num_k, hd, eps, &out_shape,
        ).unwrap();
        dev.synchronize();
        let mut t3 = Vec::new();
        for _ in 0..3 {
            let s_tmp = dev.zeros(&Shape::new(vec![num_v, hd, hd]), DType::F32).unwrap();
            let t0 = Instant::now();
            let _ = dev.kda_gated_delta_rule_scan(
                conv_d.as_ref(), alpha_d.as_ref(), beta_d.as_ref(),
                dt_b_d.as_ref(), ssm_a_d.as_ref(), nw_d.as_ref(), None,
                s_tmp.as_ref(), seq_len, num_v, num_k, hd, eps, &out_shape,
            ).unwrap();
            dev.synchronize();
            t3.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        unsafe { std::env::set_var("GRIM_KDA_SCAN_CHUNKED", "1"); }
        let (out_v4, _) = dev.kda_gated_delta_rule_scan(
            conv_d.as_ref(), alpha_d.as_ref(), beta_d.as_ref(),
            dt_b_d.as_ref(), ssm_a_d.as_ref(), nw_d.as_ref(), None,
            s_v4.as_ref(), seq_len, num_v, num_k, hd, eps, &out_shape,
        ).unwrap();
        dev.synchronize();
        let mut t4 = Vec::new();
        for _ in 0..3 {
            let s_tmp = dev.zeros(&Shape::new(vec![num_v, hd, hd]), DType::F32).unwrap();
            let t0 = Instant::now();
            let _ = dev.kda_gated_delta_rule_scan(
                conv_d.as_ref(), alpha_d.as_ref(), beta_d.as_ref(),
                dt_b_d.as_ref(), ssm_a_d.as_ref(), nw_d.as_ref(), None,
                s_tmp.as_ref(), seq_len, num_v, num_k, hd, eps, &out_shape,
            ).unwrap();
            dev.synchronize();
            t4.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let o3 = out_v3.to_cpu_vec_f32().unwrap();
        let o4 = out_v4.to_cpu_vec_f32().unwrap();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in o3.iter().zip(o4.iter()) {
            let d = (*a as f64) - (*b as f64);
            num += d * d;
            den += (*a as f64) * (*a as f64);
        }
        let rel_l2 = (num / den.max(1e-30)).sqrt();
        let v3_ms = median(t3);
        let v4_ms = median(t4);
        eprintln!(
            "[gdn-v4-wy32 @ {seq_len}] v3={v3_ms:.2} ms (48L={:.1} ms), v4={v4_ms:.2} ms (48L={:.1} ms), speedup={:.2}x, rel_L2={rel_l2:.4e}",
            v3_ms * 48.0,
            v4_ms * 48.0,
            v3_ms / v4_ms
        );
    }
    unsafe { std::env::remove_var("GRIM_KDA_SCAN_CHUNKED"); }
}

#[test]
#[ignore]
fn test_wmma_i8_q80_q81_doublek_parity_and_speed() {
    use grim_backend_rocm::BackendStorage;
    let dev = RocmDevice::new(0);
    // 1. Correctness + bitwise determinism at M=128, N=256, K=512
    let (m, n, k) = (128usize, 256usize, 512usize);
    let nb = k / 32;
    let mut rng = 0xfeed_face_cafe_beefu64;
    let mut next_f = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let a_f32: Vec<f32> = (0..m * k).map(|_| next_f()).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| next_f()).collect();

    // Quantize A to Q8_1 (36 B/block) and B to Q8_0 (34 B/block) on CPU
    let mut a_q81 = vec![0u8; m * nb * 36];
    let mut a_deq = vec![0.0f32; m * k];
    for r in 0..m {
        for blk in 0..nb {
            let base = r * k + blk * 32;
            let amax = a_f32[base..base + 32].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let dh = half::f16::from_f32(d);
            let df = dh.to_f32();
            let off = (r * nb + blk) * 36;
            a_q81[off..off + 2].copy_from_slice(&dh.to_le_bytes());
            let mut sum = 0.0f32;
            for i in 0..32 {
                let q = if df > 0.0 { (a_f32[base + i] / df).round().clamp(-127.0, 127.0) as i8 } else { 0i8 };
                a_q81[off + 4 + i] = q as u8;
                let v = (q as f32) * df;
                a_deq[base + i] = v;
                sum += v;
            }
            let sh = half::f16::from_f32(sum);
            a_q81[off + 2..off + 4].copy_from_slice(&sh.to_le_bytes());
        }
    }

    let mut b_q80 = vec![0u8; n * nb * 34];
    let mut b_deq = vec![0.0f32; n * k];
    for r in 0..n {
        for blk in 0..nb {
            let base = r * k + blk * 32;
            let amax = b_f32[base..base + 32].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let dh = half::f16::from_f32(d);
            let df = dh.to_f32();
            let off = (r * nb + blk) * 34;
            b_q80[off..off + 2].copy_from_slice(&dh.to_le_bytes());
            for i in 0..32 {
                let q = if df > 0.0 { (b_f32[base + i] / df).round().clamp(-127.0, 127.0) as i8 } else { 0i8 };
                b_q80[off + 2 + i] = q as u8;
                b_deq[base + i] = (q as f32) * df;
            }
        }
    }

    // CPU exact oracle on dequantized Q8_1 x Q8_0
    let mut c_ref = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += (a_deq[i * k + t] as f64) * (b_deq[j * k + t] as f64);
            }
            c_ref[i * n + j] = acc as f32;
        }
    }

    let a_dev = grim_backend_rocm::RocmStorage::copy_from_host_raw_bytes(&a_q81, &Shape::new(vec![m, nb * 36]), DType::U8, &dev.allocator_handle(), 0).unwrap();
    let b_dev = grim_backend_rocm::RocmStorage::copy_from_host_raw_bytes(&b_q80, &Shape::new(vec![n, nb * 34]), DType::U8, &dev.allocator_handle(), 0).unwrap();
    let c_dev1 = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &dev.allocator_handle(), 0).unwrap();
    let c_dev2 = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &dev.allocator_handle(), 0).unwrap();

    dev.launch_wmma_i8_q80_q81_doublek_gfx12(&a_dev, &b_dev, &c_dev1, m, n, k).unwrap();
    dev.launch_wmma_i8_q80_q81_doublek_gfx12(&a_dev, &b_dev, &c_dev2, m, n, k).unwrap();
    dev.synchronize();

    let c1 = c_dev1.to_cpu_vec_f32().unwrap();
    let c2 = c_dev2.to_cpu_vec_f32().unwrap();
    let bit_eq = c1.iter().zip(c2.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
    let max_abs = c1.iter().zip(c_ref.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("[1C wmma-i8-doublek parity {m}x{n}x{k}] max_abs={max_abs:.4e}, bit_eq={bit_eq}");
    assert!(bit_eq, "wmma_i8_doublek not bitwise deterministic!");
    assert!(max_abs < 1e-3, "wmma_i8_doublek max_abs too high: {max_abs}");

    // 2. Speed bench at Qwen3.8-27B prefill shape M=2048, N=5120, K=5120
    let (bm, bn, bk) = (2048usize, 5120usize, 5120usize);
    let bnb = bk / 32;
    let ba = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![bm, bnb * 36]), DType::U8, &dev.allocator_handle(), 0).unwrap();
    let bb = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![bn, bnb * 34]), DType::U8, &dev.allocator_handle(), 0).unwrap();
    let bc = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![bm, bn]), DType::F32, &dev.allocator_handle(), 0).unwrap();
    dev.launch_wmma_i8_q80_q81_doublek_gfx12(&ba, &bb, &bc, bm, bn, bk).unwrap();
    dev.synchronize();
    let mut times = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        dev.launch_wmma_i8_q80_q81_doublek_gfx12(&ba, &bb, &bc, bm, bn, bk).unwrap();
        dev.synchronize();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let ms = median(times);
    let flops = 2.0 * (bm as f64) * (bn as f64) * (bk as f64);
    let tops = flops / (ms * 1e-3) / 1e12;
    eprintln!("[1C wmma-i8-doublek perf {bm}x{bn}x{bk}] {ms:.3} ms ({tops:.2} TOP/s)");
}

/// Phase 2B: Verify F32 KV attention (prefill + decode flash_decode) is bitwise deterministic.
#[test]
#[ignore]
fn test_f32_kv_attention_determinism() {
    use grim_backend_rocm::AttentionOps;
    let dev = RocmDevice::new(0);
    let (nh, nkv, hd) = (24usize, 4usize, 256usize);
    let kv_len = 2048usize;
    let mut rng = 0xabcd_1234_ef56_7890u64;
    let mut next_f = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 40) as f32 / (1u64 << 24) as f32) * 0.5 - 0.25
    };
    let k_h: Vec<f32> = (0..kv_len * nkv * hd).map(|_| next_f()).collect();
    let v_h: Vec<f32> = (0..kv_len * nkv * hd).map(|_| next_f()).collect();
    let k_d = dev.from_cpu(&k_h, &Shape::new(vec![kv_len, nkv, hd]), DType::F32).unwrap();
    let v_d = dev.from_cpu(&v_h, &Shape::new(vec![kv_len, nkv, hd]), DType::F32).unwrap();

    for &sl in &[1usize, 128] {
        let q_h: Vec<f32> = (0..sl * nh * hd).map(|_| next_f()).collect();
        let q_d = dev.from_cpu(&q_h, &Shape::new(vec![sl, nh, hd]), DType::F32).unwrap();
        let out_shape = Shape::new(vec![sl, nh, hd]);
        let (o1, _) = dev.qkv_attention(q_d.as_ref(), k_d.as_ref(), v_d.as_ref(), nkv, kv_len, (kv_len - sl) as u32, None, &out_shape, None, None).unwrap();
        let (o2, _) = dev.qkv_attention(q_d.as_ref(), k_d.as_ref(), v_d.as_ref(), nkv, kv_len, (kv_len - sl) as u32, None, &out_shape, None, None).unwrap();
        dev.synchronize();
        let v1 = o1.to_cpu_vec_f32().unwrap();
        let v2 = o2.to_cpu_vec_f32().unwrap();
        let bit_eq = v1.iter().zip(v2.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        let max_diff = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("[2B f32-kv-det sl={sl} kv={kv_len}] bit_eq={bit_eq}, max_diff={max_diff:.4e}");
        assert!(bit_eq, "F32 KV attention nondeterministic at sl={sl}: max_diff={max_diff}");
    }
}

/// Phase 4: INT8 epilogue conformance check (res8/2609.00363).
/// Verifies boundary int8 codes (-128, -127, 0, 127) accumulate in i32 without
/// wraparound and match exact i64/f64 reference bitwise-deterministically.
#[test]
#[ignore]
fn test_int8_epilogue_conformance() {
    use grim_backend_rocm::BackendStorage;
    let dev = RocmDevice::new(0);
    let (m, n, k) = (64usize, 64usize, 128usize);
    let nb = k / 32;
    let boundary_vals: [i8; 4] = [-128, -127, 0, 127];

    let mut a_q81 = vec![0u8; m * nb * 36];
    let mut b_q80 = vec![0u8; n * nb * 34];
    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; n * k];
    let scale_h = half::f16::from_f32(0.0625); // exact power-of-two scale (1/16)
    let scale_f = scale_h.to_f32();

    for r in 0..m {
        for blk in 0..nb {
            let off = (r * nb + blk) * 36;
            a_q81[off..off + 2].copy_from_slice(&scale_h.to_le_bytes());
            for i in 0..32 {
                let v = boundary_vals[(r + blk + i) & 3];
                a_q81[off + 4 + i] = v as u8;
                a_i8[r * k + blk * 32 + i] = v;
            }
        }
    }
    for r in 0..n {
        for blk in 0..nb {
            let off = (r * nb + blk) * 34;
            b_q80[off..off + 2].copy_from_slice(&scale_h.to_le_bytes());
            for i in 0..32 {
                let v = boundary_vals[(r * 3 + blk + i) & 3];
                b_q80[off + 2 + i] = v as u8;
                b_i8[r * k + blk * 32 + i] = v;
            }
        }
    }

    let a_dev = grim_backend_rocm::RocmStorage::copy_from_host_raw_bytes(&a_q81, &Shape::new(vec![m, nb * 36]), DType::U8, &dev.allocator_handle(), 0).unwrap();
    let b_dev = grim_backend_rocm::RocmStorage::copy_from_host_raw_bytes(&b_q80, &Shape::new(vec![n, nb * 34]), DType::U8, &dev.allocator_handle(), 0).unwrap();
    let c_dev1 = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &dev.allocator_handle(), 0).unwrap();
    let c_dev2 = grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &dev.allocator_handle(), 0).unwrap();

    dev.launch_wmma_i8_q80_q81_doublek_gfx12(&a_dev, &b_dev, &c_dev1, m, n, k).unwrap();
    dev.launch_wmma_i8_q80_q81_doublek_gfx12(&a_dev, &b_dev, &c_dev2, m, n, k).unwrap();
    dev.synchronize();

    let c1 = c_dev1.to_cpu_vec_f32().unwrap();
    let c2 = c_dev2.to_cpu_vec_f32().unwrap();
    assert!(c1.iter().zip(c2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "INT8 epilogue not bitwise deterministic");

    let mut max_abs = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut dot: i64 = 0;
            for t in 0..k {
                dot += (a_i8[i * k + t] as i64) * (b_i8[j * k + t] as i64);
            }
            let expected = (dot as f32) * (scale_f * scale_f);
            max_abs = max_abs.max((c1[i * n + j] - expected).abs());
        }
    }
    eprintln!("[Phase 4 int8-epilogue-conformance] max_abs={max_abs:.4e} (exact power-of-two scale)");
    assert_eq!(max_abs, 0.0, "INT8 epilogue conformance mismatch: {max_abs}");
}
