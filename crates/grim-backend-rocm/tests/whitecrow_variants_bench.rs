//! Benchmark WhiteCrow (W4A4 OSTQuant / Sudot8 GEMV) variants:
//! - Variant 0 (Baseline): `grim_dot8_w4a4_gemv_v5`
//! - Variant 1 (Dual Row): `grim_dot8_w4a4_gemv_rows_r2` (at M=2)

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
fn bench_whitecrow_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    if !dev.gcn_arch().starts_with("gfx12") {
        println!("Requires gfx12 (RDNA4), skipping.");
        return Ok(());
    }

    println!("==========================================================================================");
    println!("WHITECROW (W4A4 SUDOT8 GEMV) VARIANT BENCHMARK (N=17408, K=5120)");
    println!("==========================================================================================");

    let (n, k) = (17408, 5120);
    let n_groups = k / 128;
    let words_per_col = k / 8;

    for m in [1, 2] {
        let mut rng = Lcg(0x6E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        let (qw_bytes, sc_bytes, zr_bytes) = grim_quant::quant_ostquant_w4_group128(&w, n, k)
            .map_err(|e| format!("crow quant: {e}"))?;

        let b_qw_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &qw_bytes,
            &Shape::new(vec![n, words_per_col]),
            DType { arith: ArithType::U32, storage: Storage::Native },
        )?;

        let b_sc_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &sc_bytes,
            &Shape::new(vec![n, n_groups]),
            DType { arith: ArithType::BF16, storage: Storage::Native },
        )?;

        let b_zr_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &zr_bytes,
            &Shape::new(vec![n, n_groups]),
            DType { arith: ArithType::U8, storage: Storage::Native },
        )?;

        // Pre-quantize A activations into W4A4 format using host quantizer (matches device)
        let mut a_codes = vec![0u32; m * n_groups * 16];
        let mut a_scales = vec![0.0f32; m * n_groups];
        let mut a_sums = vec![0i32; m * n_groups];

        for row in 0..m {
            for g in 0..n_groups {
                let grp_a = &a[row * k + g * 128..row * k + (g + 1) * 128];
                let max_val = grp_a.iter().fold(0.0f32, |mm, &x| mm.max(x.abs()));
                let scale = if max_val == 0.0 { 1.0 } else { max_val / 7.0 };
                let inv = 1.0 / scale;
                a_scales[row * n_groups + g] = scale;

                let mut sum_q = 0i32;
                for (w_idx, chunk) in grp_a.chunks(8).enumerate() {
                    let mut word = 0u32;
                    for (nib_idx, &val) in chunk.iter().enumerate() {
                        let q = ((val * inv).round() as i32).clamp(-8, 7);
                        sum_q += q;
                        let u_nib = (q as u32) & 0x0F;
                        word |= u_nib << (nib_idx * 4);
                    }
                    a_codes[row * n_groups * 16 + g * 16 + w_idx] = word;
                }
                a_sums[row * n_groups + g] = sum_q;
            }
        }

        let a_codes_dev = MemoryOps::from_cpu_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(a_codes.as_ptr() as *const u8, a_codes.len() * 4) },
            &Shape::new(vec![m, n_groups * 16]),
            DType { arith: ArithType::U32, storage: Storage::Native },
        )?;

        let a_scales_dev = MemoryOps::from_cpu_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(a_scales.as_ptr() as *const u8, a_scales.len() * 4) },
            &Shape::new(vec![m, n_groups]),
            DType { arith: ArithType::F32, storage: Storage::Native },
        )?;

        let a_sums_dev = MemoryOps::from_cpu_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(a_sums.as_ptr() as *const u8, a_sums.len() * 4) },
            &Shape::new(vec![m, n_groups]),
            DType { arith: ArithType::U32, storage: Storage::Native },
        )?;

        let y_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; m * n * 4],
            &Shape::new(vec![m, n]),
            DType { arith: ArithType::F32, storage: Storage::Native },
        )?;

        let a_c = as_rocm(&*a_codes_dev).unwrap();
        let a_s = as_rocm(&*a_scales_dev).unwrap();
        let a_sum = as_rocm(&*a_sums_dev).unwrap();
        let b_qw = as_rocm(&*b_qw_dev).unwrap();
        let b_sc = as_rocm(&*b_sc_dev).unwrap();
        let b_zr = as_rocm(&*b_zr_dev).unwrap();
        let y_st = as_rocm(&*y_dev).unwrap();

        // 1. Run Baseline v5
        dev.launch_dot8_w4a4_gemv_v5_for_ab(a_c, a_s, a_sum, b_qw, b_sc, b_zr, y_st, m, n, k)?;
        dev.synchronize();
        let res_base = y_st.copy_to_host()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();

        let mut diff_r2 = 0;
        let mut max_abs_r2 = 0.0f32;
        if m == 2 {
            dev.launch_dot8_w4a4_gemv_rows_r2_for_ab(a_c, a_s, a_sum, b_qw, b_sc, b_zr, y_st, m, n, k)?;
            dev.synchronize();
            let res_r2 = y_st.copy_to_host()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
            for (g, x) in res_r2.iter().zip(&res_base) {
                let d = (g - x).abs();
                if d > 0.0 {
                    diff_r2 += 1;
                    max_abs_r2 = max_abs_r2.max(d);
                }
            }
            println!("\nParity Checks for M={m}:");
            println!("  rows_r2 vs baseline v5: diff_count = {diff_r2} (max_abs={max_abs_r2:e})");
        }

        let iters = 50;

        // Timing Baseline v5
        for _ in 0..5 {
            dev.launch_dot8_w4a4_gemv_v5_for_ab(a_c, a_s, a_sum, b_qw, b_sc, b_zr, y_st, m, n, k)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_dot8_w4a4_gemv_v5_for_ab(a_c, a_s, a_sum, b_qw, b_sc, b_zr, y_st, m, n, k)?;
        }
        dev.synchronize();
        let base_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let mut r2_us = 0.0;
        if m == 2 {
            for _ in 0..5 {
                dev.launch_dot8_w4a4_gemv_rows_r2_for_ab(a_c, a_s, a_sum, b_qw, b_sc, b_zr, y_st, m, n, k)?;
            }
            dev.synchronize();
            let t0 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot8_w4a4_gemv_rows_r2_for_ab(a_c, a_s, a_sum, b_qw, b_sc, b_zr, y_st, m, n, k)?;
            }
            dev.synchronize();
            r2_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        }

        let total_bytes = (qw_bytes.len() + sc_bytes.len() + zr_bytes.len()) as f64;
        println!("\nLatency breakdown for M={m}:");
        println!("---------------------------------------------------------------------------------------------------");
        println!("Variant                                | Latency (us) | Speedup vs Base | Eff. GB/s | Per-Token us");
        println!("---------------------------------------------------------------------------------------------------");
        println!("Baseline (v5 sudot8)                   | {:>10.2} us | 1.00x           | {:>8.1}  | {:>10.2} us", base_us, total_bytes / 1e9 / (base_us * 1e-6), base_us / m as f64);
        if m == 2 {
            println!("Variant 1 (rows_r2 dual row reuse)     | {:>10.2} us | {:>4.2}x           | {:>8.1}  | {:>10.2} us", r2_us, base_us / r2_us, total_bytes / 1e9 / (r2_us * 1e-6), r2_us / m as f64);
        }
        println!("===================================================================================================");
    }

    Ok(())
}
