//! Benchmark comparing Q3_K variants on a 60 MB weight tensor:
//! - Baseline: `grim_dot4_q3k_q81_gemv_sub_v2`
//! - M=1 Vectorized: `grim_dot4_q3k_q81_gemv_sub_v4_vec`
//! - M=2 Vectorized Dual Row: `grim_dot4_q3k_q81_gemv_rows_r2_vec`

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
fn bench_q3k_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("Q3_K INT8 DOT4 GEMV VARIANT BENCHMARK (60 MB WEIGHT TENSOR: N=28672, K=5120)");
    println!("==========================================================================================");

    // K=5120: 20 super-blocks (256 weights, 110 bytes) -> 2200 bytes/column.
    // N=28672 -> 28672 * 2200 = 63,078,400 bytes = 60.16 MB weight tensor.
    // Sized to > 60 MB to completely bypass Infinity Cache (L3).
    let (n, k) = (28672, 5120);

    for m in [1, 2] {
        let mut rng = Lcg(0x3E67 ^ (m as u64));
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

        // 2. Run M=1 Vectorized Var 1 (v4_vec)
        dev.launch_dot4_q3k_q81_gemv_sub_v4_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_v4 = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();

        let mut diff_v4 = 0;
        let mut max_abs_v4 = 0.0f32;
        for (g, x) in res_v4.iter().zip(&res_base) {
            let d = (g - x).abs();
            if d > 0.0 {
                diff_v4 += 1;
                max_abs_v4 = max_abs_v4.max(d);
            }
        }
        println!("\nParity Checks for M={m}:");
        println!("  sub_v4_vec vs baseline: diff_count = {diff_v4} (max_abs={max_abs_v4:e})");
        assert_eq!(diff_v4, 0, "sub_v4_vec parity check");

        if m == 2 {
            dev.launch_dot4_q3k_q81_gemv_rows_r2_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
            dev.synchronize();
            let res_r2 = y_st
                .copy_to_host()?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            let mut diff_r2 = 0;
            let mut max_abs_r2 = 0.0f32;
            for (g, x) in res_r2.iter().zip(&res_base) {
                let d = (g - x).abs();
                if d > 0.0 {
                    diff_r2 += 1;
                    max_abs_r2 = max_abs_r2.max(d);
                }
            }
            println!("  rows_r2_vec vs baseline: diff_count = {diff_r2} (max_abs={max_abs_r2:e})");
            assert_eq!(diff_r2, 0, "rows_r2_vec parity check");
        }

        let iters = 40;

        // Timing Baseline
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

        // Timing v4_vec
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

        // Q3_K weight bytes = (N * K / 256) * 110
        let q3k_bytes = ((n * k) / 256) * 110;
        let bw_base = (q3k_bytes as f64 * m as f64) / (dur_base * 1e-6) / 1e9;
        let bw_v4 = (q3k_bytes as f64 * m as f64) / (dur_v4 * 1e-6) / 1e9;

        println!("\nLatency & Bandwidth for M={m}:");
        println!("  Baseline (sub_v2):          {dur_base:6.2} us | eff BW: {bw_base:6.1} GB/s");
        println!(
            "  Variant 1 (sub_v4_vec):     {dur_v4:6.2} us | eff BW: {bw_v4:6.1} GB/s (speedup: {:4.2}x)",
            dur_base / dur_v4
        );

        if m == 2 {
            for _ in 0..5 {
                dev.launch_dot4_q3k_q81_gemv_rows_r2_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t2 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_q3k_q81_gemv_rows_r2_vec_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let dur_r2 = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);
            let bw_r2 = (q3k_bytes as f64 * m as f64) / (dur_r2 * 1e-6) / 1e9;
            println!(
                "  Variant 2 (rows_r2_vec):    {dur_r2:6.2} us | eff BW: {bw_r2:6.1} GB/s (speedup: {:4.2}x total, {:6.2} us/tok)",
                dur_base / dur_r2,
                dur_r2 / 2.0
            );
        }
        println!("------------------------------------------------------------------------------------------");
    }

    Ok(())
}
