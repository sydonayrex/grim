//! Benchmark comparing all Q3_K M=1 improvement variations on a 60 MB weight tensor:
//! - Baseline: `grim_dot4_q3k_q81_gemv_sub_v2` (scalar loads, 4 cols)
//! - Var 1:    `grim_dot4_q3k_q81_gemv_sub_v4_vec` (128-bit vector loads, 4 cols)
//! - Var 2:    `grim_dot4_q3k_q81_gemv_sub_v4_vec_prefetch` (128-bit vector + prefetch)
//! - Var 3:    `grim_dot4_q3k_q81_gemv_sub_v6_vec_col8` (128-bit vector + 8 cols)

use grim_backend_rocm::{as_rocm, RocmDevice};
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};
use std::time::Instant;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

#[test]
fn bench_q3k_m1_variations() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("Q3_K M=1 COMPREHENSIVE VARIANT BENCHMARK (60 MB WEIGHT TENSOR: N=28672, K=5120)");
    println!("==========================================================================================");

    let (m, n, k) = (1, 28672, 5120);

    let mut rng = Lcg(0x3E67 ^ 1);
    let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
    let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

    let w_q3k = grim_quant::quant_q3k(&w).expect("quant_q3k");

    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )?;

    let a_q81_bytes = m * (k / 32) * 36;
    let a_q81_t = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; a_q81_bytes],
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;

    let a_f32_st = as_rocm(&*a_t).unwrap();
    let a_st = as_rocm(&*a_q81_t).unwrap();
    dev.launch_quantize_q8_1(a_f32_st, a_st, m, k)?;

    let w_t = MemoryOps::from_cpu_bytes(
        &dev,
        &w_q3k,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;

    let y_t = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; m * n * 4],
        &Shape::new(vec![m, n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )?;

    let w_st = as_rocm(&*w_t).unwrap();
    let y_st = as_rocm(&*y_t).unwrap();

    // 1. Run Baseline sub_v2
    dev.launch_dot4_q3k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
    dev.synchronize();
    let res_base = y_st
        .copy_to_host()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>();

    // 2. Parity check Var 1: sub_v4_vec
    dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
    dev.synchronize();
    let res_v4 = y_st
        .copy_to_host()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>();
    let mut diff_v4 = 0;
    for (g, x) in res_v4.iter().zip(&res_base) {
        if (g - x).abs() > 0.0 {
            diff_v4 += 1;
        }
    }
    println!("Parity Check Var 1 (sub_v4_vec): diff_count = {diff_v4}");
    assert_eq!(diff_v4, 0, "sub_v4_vec parity check");

    // 3. Parity check Var 2: sub_v4_vec_prefetch
    dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
    dev.synchronize();
    let res_pref = y_st
        .copy_to_host()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>();
    let mut diff_pref = 0;
    for (g, x) in res_pref.iter().zip(&res_base) {
        if (g - x).abs() > 0.0 {
            diff_pref += 1;
        }
    }
    println!("Parity Check Var 2 (sub_v4_vec_prefetch): diff_count = {diff_pref}");
    assert_eq!(diff_pref, 0, "sub_v4_vec_prefetch parity check");

    // 4. Parity check Var 3: sub_v6_vec_col8
    dev.launch_dot4_q3k_q81_gemv_sub_v6_vec_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
    dev.synchronize();
    let res_col8 = y_st
        .copy_to_host()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>();
    let mut diff_col8 = 0;
    for (g, x) in res_col8.iter().zip(&res_base) {
        if (g - x).abs() > 0.0 {
            diff_col8 += 1;
        }
    }
    println!("Parity Check Var 3 (sub_v6_vec_col8): diff_count = {diff_col8}");
    assert_eq!(diff_col8, 0, "sub_v6_vec_col8 parity check");

    let iters = 40;

    // Benchmark Baseline
    for _ in 0..5 {
        dev.launch_dot4_q3k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        dev.launch_dot4_q3k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let dur_base = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    // Benchmark Var 1: sub_v4_vec
    for _ in 0..5 {
        dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let t1 = Instant::now();
    for _ in 0..iters {
        dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let dur_v4 = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    // Benchmark Var 2: sub_v4_vec_prefetch
    for _ in 0..5 {
        dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let t2 = Instant::now();
    for _ in 0..iters {
        dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let dur_pref = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    // Benchmark Var 3: sub_v6_vec_col8
    for _ in 0..5 {
        dev.launch_dot4_q3k_q81_gemv_sub_v6_vec_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let t3 = Instant::now();
    for _ in 0..iters {
        dev.launch_dot4_q3k_q81_gemv_sub_v6_vec_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
    }
    dev.synchronize();
    let dur_col8 = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

    let q3k_bytes = ((n * k) / 256) * 110;
    let bw = |dur: f64| (q3k_bytes as f64) / (dur * 1e-6) / 1e9;

    println!("\nLatency & Bandwidth for M=1 on 60 MB tensor:");
    println!("--------------------------------------------------------------------------------------------------");
    println!("Variant                                        | Latency (us) | Eff. GB/s | Speedup vs Base");
    println!("--------------------------------------------------------------------------------------------------");
    println!("Baseline (sub_v2: 4 cols, scalar loads)        |    {dur_base:6.2} us |  {:6.1}   | 1.00x", bw(dur_base));
    println!("Var 1 (sub_v4_vec: 4 cols, 128-bit vector)     |    {dur_v4:6.2} us |  {:6.1}   | {:4.2}x", bw(dur_v4), dur_base / dur_v4);
    println!("Var 2 (sub_v4_vec_prefetch: vec + prefetch)    |    {dur_pref:6.2} us |  {:6.1}   | {:4.2}x", bw(dur_pref), dur_base / dur_pref);
    println!("Var 3 (sub_v6_vec_col8: 8 cols, 128-bit vector)|    {dur_col8:6.2} us |  {:6.1}   | {:4.2}x", bw(dur_col8), dur_base / dur_col8);
    println!("==================================================================================================");

    Ok(())
}
