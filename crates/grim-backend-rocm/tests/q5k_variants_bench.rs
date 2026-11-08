//! Benchmark comparing Q5_K variants:
//! - Variant 0 (Baseline): `grim_dot4_q5k_q81_gemv_sub_v2`
//! - Variant 1 (Prefetch): `grim_dot4_q5k_q81_gemv_sub_v3_prefetch`
//! - Variant 2 (Dual Row): `grim_dot4_q5k_q81_gemv_rows_r2` (at M=2)

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
fn bench_q5k_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("Q5_K INT8 DOT4 GEMV VARIANT BENCHMARK (60 MB WEIGHT TENSOR: N=17920, K=5120)");
    println!("==========================================================================================");

    // K=5120: 20 super-blocks (256 weights, 176 bytes) -> 3520 bytes/column.
    // N=17920 -> 17920 * 3520 = 63,078,400 bytes = 60.16 MB weight tensor.
    // Exceeds / completely saturates the 60 MB footprint to eliminate cache skew.
    let (n, k) = (17920, 5120);

    for m in [1, 2] {
        let mut rng = Lcg(0x5E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        let w_q5k = grim_quant::quant_q5k(&w).expect("quant_q5k");

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
            &w_q5k,
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
        dev.launch_dot4_q5k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_base = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();

        // 2. Run Variant 1 (v3_prefetch)
        dev.launch_dot4_q5k_q81_gemv_sub_v3_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_v3 = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();

        // Parity check v3 vs baseline
        let mut diff_v3 = 0;
        let mut max_abs_v3 = 0.0f32;
        for (g, x) in res_v3.iter().zip(&res_base) {
            let d = (g - x).abs();
            if d > 0.0 {
                diff_v3 += 1;
                max_abs_v3 = max_abs_v3.max(d);
            }
        }
        println!("\nParity Checks for M={m}:");
        println!("  v3_prefetch vs baseline: diff_count = {diff_v3} (max_abs={max_abs_v3:e})");

        let mut diff_r2 = 0;
        let mut max_abs_r2 = 0.0f32;
        if m == 2 {
            dev.launch_dot4_q5k_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            dev.synchronize();
            let res_r2 = y_st
                .copy_to_host()?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            for (g, x) in res_r2.iter().zip(&res_base) {
                let d = (g - x).abs();
                if d > 0.0 {
                    diff_r2 += 1;
                    max_abs_r2 = max_abs_r2.max(d);
                }
            }
            println!("  rows_r2 vs baseline: diff_count = {diff_r2} (max_abs={max_abs_r2:e})");
        }

        let iters = 40;

        // Timing Baseline
        for _ in 0..5 {
            dev.launch_dot4_q5k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_q5k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_base = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing v3_prefetch
        for _ in 0..5 {
            dev.launch_dot4_q5k_q81_gemv_sub_v3_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t1 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_q5k_q81_gemv_sub_v3_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_v3 = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Q5_K weight bytes = (N * K / 256) * 176
        let q5k_bytes = ((n * k) / 256) * 176;
        let bw_base = (q5k_bytes as f64 * m as f64) / (dur_base * 1e-6) / 1e9;
        let bw_v3 = (q5k_bytes as f64 * m as f64) / (dur_v3 * 1e-6) / 1e9;

        println!("\nLatency & Bandwidth for M={m}:");
        println!("  Baseline (sub_v2):          {dur_base:6.2} us | eff BW: {bw_base:6.1} GB/s");
        println!(
            "  Variant 1 (v3_prefetch):    {dur_v3:6.2} us | eff BW: {bw_v3:6.1} GB/s (speedup: {:4.2}x)",
            dur_base / dur_v3
        );

        if m == 2 {
            for _ in 0..5 {
                dev.launch_dot4_q5k_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t2 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_q5k_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let dur_r2 = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);
            let bw_r2 = (q5k_bytes as f64 * m as f64) / (dur_r2 * 1e-6) / 1e9;
            println!(
                "  Variant 2 (rows_r2):        {dur_r2:6.2} us | eff BW: {bw_r2:6.1} GB/s (speedup: {:4.2}x total, {:4.2}x per-token: {:6.2} us/tok)",
                dur_base / dur_r2,
                (dur_base / 2.0) / (dur_r2 / 2.0) * (dur_base / dur_r2),
                dur_r2 / 2.0
            );
        }
        println!("------------------------------------------------------------------------------------------");
    }

    Ok(())
}
