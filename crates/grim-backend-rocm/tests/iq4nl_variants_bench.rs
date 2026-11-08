//! Benchmark comparing IQ4_NL improvement variations on a 60 MB weight tensor:
//! - Baseline: `grim_dot4_iq4nl_q81_gemv` (byte-wide lookups)
//! - Var 1:    `grim_dot4_iq4nl_q81_gemv_fast` (word-wide fast lookup + 128-bit load)
//! - Var 2:    `grim_dot4_iq4nl_q81_gemv_prefetch` (software prefetching)
//! - Var 3:    `grim_dot4_iq4nl_q81_gemv_col8` (8-column wave tile)
//! - Var 4:    `grim_dot4_iq4nl_q81_gemv_rows_r2` (Dual-row weight reuse at M=2)

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
fn bench_iq4nl_all_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("IQ4_NL COMPREHENSIVE VARIANT BENCHMARK (60.48 MB WEIGHT TENSOR: N=10752, K=10240)");
    println!("==========================================================================================");

    let (n, k) = (10752, 10240);

    for m in [1, 2] {
        let mut rng = Lcg(0x3E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        let w_iq4nl = grim_quant::quant_iq4nl(&w).expect("quant_iq4nl");

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
            &w_iq4nl,
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

        // 1. Run Baseline
        dev.launch_dot4_iq4nl_q81_gemv_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_base = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();

        // 2. Parity check Var 1 (fast)
        dev.launch_dot4_iq4nl_q81_gemv_fast_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_fast = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();
        let mut diff_fast = 0;
        for (g, x) in res_fast.iter().zip(&res_base) {
            if (g - x).abs() > 0.0 {
                diff_fast += 1;
            }
        }
        println!("\nParity Checks for M={m}:");
        println!("  fast vs baseline: diff_count = {diff_fast}");
        assert_eq!(diff_fast, 0, "fast parity check");

        // 3. Parity check Var 2 (prefetch)
        dev.launch_dot4_iq4nl_q81_gemv_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
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
        println!("  prefetch vs baseline: diff_count = {diff_pref}");
        assert_eq!(diff_pref, 0, "prefetch parity check");

        // 4. Parity check Var 3 (col8)
        dev.launch_dot4_iq4nl_q81_gemv_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
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
        println!("  col8 vs baseline: diff_count = {diff_col8}");
        assert_eq!(diff_col8, 0, "col8 parity check");

        if m == 2 {
            dev.launch_dot4_iq4nl_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
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
            dev.launch_dot4_iq4nl_q81_gemv_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4nl_q81_gemv_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_base = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing Fast
        for _ in 0..5 {
            dev.launch_dot4_iq4nl_q81_gemv_fast_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t1 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4nl_q81_gemv_fast_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_fast = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing Prefetch
        for _ in 0..5 {
            dev.launch_dot4_iq4nl_q81_gemv_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t2 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4nl_q81_gemv_prefetch_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_pref = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing Col8
        for _ in 0..5 {
            dev.launch_dot4_iq4nl_q81_gemv_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t3 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4nl_q81_gemv_col8_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_col8 = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let iq4nl_bytes = (n * k / 32) * 18;
        let bw = |dur: f64| (iq4nl_bytes as f64 * m as f64) / (dur * 1e-6) / 1e9;

        println!("\nLatency & Bandwidth for M={m}:");
        println!("  Baseline (byte-wide lookup): {dur_base:6.2} us | eff BW: {:6.1} GB/s", bw(dur_base));
        println!("  Var 1 (fast word lookup):    {dur_fast:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x)", bw(dur_fast), dur_base / dur_fast);
        println!("  Var 2 (fast + prefetch):     {dur_pref:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x)", bw(dur_pref), dur_base / dur_pref);
        println!("  Var 3 (fast + 8-col tile):   {dur_col8:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x)", bw(dur_col8), dur_base / dur_col8);

        if m == 2 {
            for _ in 0..5 {
                dev.launch_dot4_iq4nl_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t4 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq4nl_q81_gemv_rows_r2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let dur_r2 = t4.elapsed().as_secs_f64() * 1e6 / (iters as f64);
            println!(
                "  Var 4 (dual-row weight reuse):{dur_r2:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x total, {:6.2} us/tok)",
                bw(dur_r2),
                dur_base / dur_r2,
                dur_r2 / 2.0
            );
        }
        println!("------------------------------------------------------------------------------------------");
    }

    Ok(())
}
