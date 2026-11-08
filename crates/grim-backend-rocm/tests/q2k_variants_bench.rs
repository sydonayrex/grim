//! Benchmark comparing Q2_K wired production variants on a 60 MB weight tensor:
//! - Baseline: `grim_dot4_q2k_q81_gemv_sub_v2`
//! - M=1 Optimal: `grim_dot4_q2k_q81_gemv_sub_v6_vec_col8` (8-column wave tile)
//! - M=2 Optimal: `grim_dot4_q2k_q81_gemv_rows_r2` (Dual-row weight reuse)

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
fn bench_q2k_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("Q2_K PRODUCTION VARIANT BENCHMARK (60 MB WEIGHT TENSOR: N=37632, K=5120)");
    println!("==========================================================================================");

    let (n, k) = (37632, 5120);

    for m in [1, 2] {
        let mut rng = Lcg(0x3E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        let w_q2k = grim_quant::quant_q2k(&w).expect("quant_q2k");

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
            &w_q2k,
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
        dev.launch_dot4_q2k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_base = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();

        // 2. Parity check Var 3 (col8)
        dev.launch_dot4_q2k_q81_gemv_sub_v6_vec_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
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
        println!("\nParity Checks for M={m}:");
        println!("  sub_v6_vec_col8 vs baseline: diff_count = {diff_col8}");
        assert_eq!(diff_col8, 0, "sub_v6_vec_col8 parity check");

        if m == 2 {
            dev.launch_dot4_q2k_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            dev.synchronize();
            let res_r2 = y_st
                .copy_to_host()?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            let mut diff_r2 = 0;
            for (g, x) in res_r2.iter().zip(&res_base) {
                if (g - x).abs() > 0.0 {
                    diff_r2 += 1;
                }
            }
            println!("  rows_r2 vs baseline: diff_count = {diff_r2}");
            assert_eq!(diff_r2, 0, "rows_r2 parity check");
        }

        let iters = 40;

        // Timing Baseline
        for _ in 0..5 {
            dev.launch_dot4_q2k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_q2k_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_base = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing Var 3: col8
        for _ in 0..5 {
            dev.launch_dot4_q2k_q81_gemv_sub_v6_vec_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t3 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_q2k_q81_gemv_sub_v6_vec_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_col8 = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let q2k_bytes = ((n * k) / 256) * 84;
        let bw = |dur: f64| (q2k_bytes as f64 * m as f64) / (dur * 1e-6) / 1e9;

        println!("\nLatency & Bandwidth for M={m}:");
        println!("  Baseline (sub_v2):          {dur_base:6.2} us | eff BW: {:6.1} GB/s", bw(dur_base));
        println!("  Col 8 Tile (sub_v6_vec):    {dur_col8:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x)", bw(dur_col8), dur_base / dur_col8);

        if m == 2 {
            for _ in 0..5 {
                dev.launch_dot4_q2k_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t4 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_q2k_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let dur_r2 = t4.elapsed().as_secs_f64() * 1e6 / (iters as f64);
            println!(
                "  Dual Row Reuse (rows_r2):   {dur_r2:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x total, {:6.2} us/tok)",
                bw(dur_r2),
                dur_base / dur_r2,
                dur_r2 / 2.0
            );
        }
        println!("------------------------------------------------------------------------------------------");
    }

    Ok(())
}
