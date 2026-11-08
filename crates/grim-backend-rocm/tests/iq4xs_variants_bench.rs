//! Benchmark comparing IQ4_XS variants on a 60 MB weight tensor:
//! - Shape: N=23552, K=5120 -> 23552 * (5120 / 256) * 136 = 64,061,440 bytes = 61.09 MB weight tensor.
//! - Variants tested:
//!   - `sub_v2`: baseline byte-wide nibble decode
//!   - `sub_v8`: live production word-wide nibble decode (B-first load order)
//!   - `sub_v5b`: column-refill pipelined v2
//!   - `rows_r2`: multi-row 2-token reuse
//!   - `rows_hv2`: hoisted-kv 2-token reuse

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
fn bench_iq4xs_all_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("IQ4_XS COMPREHENSIVE VARIANT BENCHMARK (61.09 MB WEIGHT TENSOR: N=23552, K=5120)");
    println!("==========================================================================================");

    // K=5120: 20 superblocks (256 weights, 136 bytes) -> 2720 bytes/col.
    // N=23552 -> 23552 * 2720 = 64,061,440 bytes = 61.09 MB weight tensor.
    let (n, k) = (23552, 5120);

    for m in [1, 2] {
        let mut rng = Lcg(0x3E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        let w_iq4xs = grim_quant::quant_iq4xs(&w).expect("quant_iq4xs");

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
            &w_iq4xs,
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
        dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_v2 = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();

        // 2. Parity check sub_v8
        dev.launch_dot4_iq4xs_q81_gemv_v8_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_v8 = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();
        let mut diff_v8 = 0;
        for (g, x) in res_v8.iter().zip(&res_v2) {
            if (g - x).abs() > 0.0 {
                diff_v8 += 1;
            }
        }
        println!("\nParity Checks for M={m}:");
        println!("  sub_v8 vs sub_v2: diff_count = {diff_v8}");
        assert_eq!(diff_v8, 0, "sub_v8 parity check");

        // 3. Parity check sub_v5b
        dev.launch_dot4_iq4xs_q81_gemv_v5b_for_ab(a_st, w_st, y_st, m, n, k)?;
        dev.synchronize();
        let res_v5b = y_st
            .copy_to_host()?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();
        let mut diff_v5b = 0;
        for (g, x) in res_v5b.iter().zip(&res_v2) {
            if (g - x).abs() > 0.0 {
                diff_v5b += 1;
            }
        }
        println!("  sub_v5b vs sub_v2: diff_count = {diff_v5b}");
        assert_eq!(diff_v5b, 0, "sub_v5b parity check");

        if m == 2 {
            dev.launch_dot4_iq4xs_q81_gemv_rows2_for_ab(a_st, w_st, y_st, m, n, k)?;
            dev.synchronize();
            let res_r2 = y_st
                .copy_to_host()?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            let mut diff_r2 = 0;
            for (g, x) in res_r2.iter().zip(&res_v2) {
                if (g - x).abs() > 0.0 {
                    diff_r2 += 1;
                }
            }
            println!("  rows_r2 vs sub_v2: diff_count = {diff_r2}");
            assert_eq!(diff_r2, 0, "rows_r2 parity check");

            dev.launch_dot4_iq4xs_q81_gemv_rows_hv2_for_ab(a_st, w_st, y_st, m, n, k)?;
            dev.synchronize();
            let res_hv2 = y_st
                .copy_to_host()?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            let mut diff_hv2 = 0;
            for (g, x) in res_hv2.iter().zip(&res_v2) {
                if (g - x).abs() > 0.0 {
                    diff_hv2 += 1;
                }
            }
            println!("  rows_hv2 vs sub_v2: diff_count = {diff_hv2}");
            assert_eq!(diff_hv2, 0, "rows_hv2 parity check");
        }

        let iters = 40;

        // Timing sub_v2
        for _ in 0..5 {
            dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_v2 = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing sub_v8
        for _ in 0..5 {
            dev.launch_dot4_iq4xs_q81_gemv_v8_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t1 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4xs_q81_gemv_v8_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_v8 = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        // Timing sub_v5b
        for _ in 0..5 {
            dev.launch_dot4_iq4xs_q81_gemv_v5b_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t2 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot4_iq4xs_q81_gemv_v5b_for_ab(a_st, w_st, y_st, m, n, k)?;
        }
        dev.synchronize();
        let dur_v5b = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let iq4xs_bytes = (n * k / 256) * 136;
        let bw = |dur: f64| (iq4xs_bytes as f64 * m as f64) / (dur * 1e-6) / 1e9;

        println!("\nLatency & Bandwidth for M={m}:");
        println!("  sub_v2 (byte-wide):         {dur_v2:6.2} us | eff BW: {:6.1} GB/s", bw(dur_v2));
        println!("  sub_v8 (word-wide, B-first):{dur_v8:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x)", bw(dur_v8), dur_v2 / dur_v8);
        println!("  sub_v5b (refill pipelined): {dur_v5b:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x)", bw(dur_v5b), dur_v2 / dur_v5b);

        if m == 2 {
            for _ in 0..5 {
                dev.launch_dot4_iq4xs_q81_gemv_rows2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t3 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq4xs_q81_gemv_rows2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let dur_r2 = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            for _ in 0..5 {
                dev.launch_dot4_iq4xs_q81_gemv_rows_hv2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t4 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq4xs_q81_gemv_rows_hv2_for_ab(a_st, w_st, y_st, m, n, k)?;
            }
            dev.synchronize();
            let dur_hv2 = t4.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            println!(
                "  rows_r2 (dual-row reuse):   {dur_r2:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x total, {:6.2} us/tok)",
                bw(dur_r2),
                dur_v2 / dur_r2,
                dur_r2 / 2.0
            );
            println!(
                "  rows_hv2 (hoisted-kv reuse):{dur_hv2:6.2} us | eff BW: {:6.1} GB/s (speedup: {:4.2}x total, {:6.2} us/tok)",
                bw(dur_hv2),
                dur_v2 / dur_hv2,
                dur_hv2 / 2.0
            );
        }
        println!("------------------------------------------------------------------------------------------");
    }

    Ok(())
}
