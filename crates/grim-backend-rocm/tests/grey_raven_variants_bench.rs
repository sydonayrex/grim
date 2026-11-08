//! Benchmark Grey Raven variants:
//! - Variant 0 (Baseline): `grim_grey_raven_b_prologue` + `grim_grey_raven_gemm_v2`
//! - Variant 1: `grim_grey_raven_b_prologue_v2` (HW-accelerated V_CVT_PK_FP8) + `grim_grey_raven_gemm_v2`
//! - Variant 2: `grim_grey_raven_b_prologue` + `grim_grey_raven_gemm_v3_prefetch`
//! - Variant 1+2: `grim_grey_raven_b_prologue_v2` + `grim_grey_raven_gemm_v3_prefetch`

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
fn bench_grey_raven_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    if !dev.gcn_arch().starts_with("gfx12") {
        println!("Requires gfx12 (RDNA4), skipping.");
        return Ok(());
    }

    println!("==========================================================================================");
    println!("GREY RAVEN (2:4 SPARSITY SWMMAC) VARIANT BENCHMARK (N=4096, K=4096)");
    println!("==========================================================================================");

    for m in [1, 16] {
        let (n, k) = (4096, 4096);
        let mut rng = Lcg(0x6E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        let pats = grim_quant::grey_raven::coupled_patterns_2_4(&w, n, k)
            .map_err(|e| format!("couple: {e}"))?;
        let blob = grim_quant::grey_raven::pack_grey_raven_hw(&w, n, k, &pats)
            .map_err(|e| format!("pack: {e}"))?;

        let a_t = MemoryOps::from_cpu_bytes(
            &dev,
            &a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &Shape::new(vec![m, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )?;

        let w_blob_t = MemoryOps::from_cpu_bytes(
            &dev,
            &blob,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Block(grim_tensor::BlockDtype::Fp8Sparse24Hw),
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

        let n_tiles = n.div_ceil(16);
        let n_windows = k.div_ceil(32);
        let m_tiles = m.div_ceil(16);

        let b_scratch = dev.white_raven_act_scratch(m_tiles * n_windows * 512)?;

        let a_st = as_rocm(&*a_t).unwrap();
        let w_st = as_rocm(&*w_blob_t).unwrap();
        let y_st = as_rocm(&*y_t).unwrap();

        // Parity verification across variants
        // 1. Run baseline
        dev.launch_grey_raven_b_prologue(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        dev.launch_grey_raven_gemm_v2_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        dev.synchronize();
        let res_base = y_st.copy_to_host()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();

        // 2. Run prologue v2 + gemm v2
        dev.launch_grey_raven_b_prologue_v2(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        dev.launch_grey_raven_gemm_v2_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        dev.synchronize();
        let res_v1 = y_st.copy_to_host()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();

        // 3. Run prologue v1 + gemm v3_prefetch
        dev.launch_grey_raven_b_prologue(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        dev.launch_grey_raven_gemm_v3_prefetch_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        dev.synchronize();
        let res_v2 = y_st.copy_to_host()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();

        // 4. Run prologue v2 + gemm v3_prefetch
        dev.launch_grey_raven_b_prologue_v2(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        dev.launch_grey_raven_gemm_v3_prefetch_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        dev.synchronize();
        let _res_v12 = y_st.copy_to_host()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();

        // Check difference between baseline and v2 (gemm v3_prefetch with same prologue should be bit identical)
        let mut diff_v2 = 0;
        let mut max_abs_v2 = 0.0f32;
        for (g, x) in res_v2.iter().zip(&res_base) {
            let d = (g - x).abs();
            if d > 0.0 {
                diff_v2 += 1;
                max_abs_v2 = max_abs_v2.max(d);
            }
        }

        // Check difference between baseline and v1 (HW CVT might differ only on exact ties / rounding mode)
        let mut diff_v1 = 0;
        let mut max_abs_v1 = 0.0f32;
        for (g, x) in res_v1.iter().zip(&res_base) {
            let d = (g - x).abs();
            if d > 0.0 {
                diff_v1 += 1;
                max_abs_v1 = max_abs_v1.max(d);
            }
        }

        println!("\nParity Checks for M={m}:");
        println!("  Variant 2 (v3_prefetch vs baseline v2): diff_count = {diff_v2} (max_abs={max_abs_v2:e})");
        println!("  Variant 1 (prologue_v2 vs baseline v2): diff_count = {diff_v1} (max_abs={max_abs_v1:e})");

        // Warmup & timing function
        let iters = 50;

        // Measure prologue v1 alone
        for _ in 0..10 {
            dev.launch_grey_raven_b_prologue(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_b_prologue(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        }
        dev.synchronize();
        let pro1_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Measure prologue v2 alone
        for _ in 0..10 {
            dev.launch_grey_raven_b_prologue_v2(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_b_prologue_v2(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
        }
        dev.synchronize();
        let pro2_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Measure GEMM v2 alone
        for _ in 0..10 {
            dev.launch_grey_raven_gemm_v2_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_gemm_v2_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let gemm_v2_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Measure GEMM v3_prefetch alone
        for _ in 0..10 {
            dev.launch_grey_raven_gemm_v3_prefetch_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_gemm_v3_prefetch_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let gemm_v3_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Measure Total End-to-End:
        // Baseline (Pro1 + GEMM v2)
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_b_prologue(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
            dev.launch_grey_raven_gemm_v2_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let total_base_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Variant 1 (Pro2 + GEMM v2)
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_b_prologue_v2(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
            dev.launch_grey_raven_gemm_v2_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let total_v1_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Variant 2 (Pro1 + GEMM v3)
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_b_prologue(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
            dev.launch_grey_raven_gemm_v3_prefetch_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let total_v2_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Variant 1+2 (Pro2 + GEMM v3)
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.launch_grey_raven_b_prologue_v2(a_st, &b_scratch, m, k, m_tiles, n_windows)?;
            dev.launch_grey_raven_gemm_v3_prefetch_for_ab(w_st, &b_scratch, y_st, m, n, k, n_tiles, n_windows)?;
        }
        dev.synchronize();
        let total_v12_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Print table
        println!("\nLatency breakdown for M={m} (4096 x 4096):");
        println!("------------------------------------------------------------------------------------------");
        println!("Component / Variant                     | Latency (us) | Speedup vs Baseline | Eff. GB/s");
        println!("------------------------------------------------------------------------------------------");
        println!("Prologue v1 (scalar E4M3 software)      | {:>10.2} us | 1.00x (base)        | {:>8.1} GB/s", pro1_us, (m * k * 4) as f64 / 1e9 / (pro1_us * 1e-6));
        println!("Prologue v2 (V_CVT_PK_FP8 vectorized)  | {:>10.2} us | {:>4.2}x              | {:>8.1} GB/s", pro2_us, pro1_us / pro2_us, (m * k * 4) as f64 / 1e9 / (pro2_us * 1e-6));
        println!("GEMM v2 (dual N-tile, 2x unroll)        | {:>10.2} us | 1.00x (base)        | {:>8.1} GB/s", gemm_v2_us, (blob.len()) as f64 / 1e9 / (gemm_v2_us * 1e-6));
        println!("GEMM v3 (dual N-tile + prefetch pipeln) | {:>10.2} us | {:>4.2}x              | {:>8.1} GB/s", gemm_v3_us, gemm_v2_us / gemm_v3_us, (blob.len()) as f64 / 1e9 / (gemm_v3_us * 1e-6));
        println!("------------------------------------------------------------------------------------------");
        println!("Total E2E Baseline (Pro1 + GEMM v2)     | {:>10.2} us | 1.00x (base)        | {:>8.1} GB/s", total_base_us, (blob.len()) as f64 / 1e9 / (total_base_us * 1e-6));
        println!("Total E2E Variant 1 (Pro2 + GEMM v2)    | {:>10.2} us | {:>4.2}x              | {:>8.1} GB/s", total_v1_us, total_base_us / total_v1_us, (blob.len()) as f64 / 1e9 / (total_v1_us * 1e-6));
        println!("Total E2E Variant 2 (Pro1 + GEMM v3)    | {:>10.2} us | {:>4.2}x              | {:>8.1} GB/s", total_v2_us, total_base_us / total_v2_us, (blob.len()) as f64 / 1e9 / (total_v2_us * 1e-6));
        println!("Total E2E Variant 1+2 (Pro2 + GEMM v3)  | {:>10.2} us | {:>4.2}x              | {:>8.1} GB/s", total_v12_us, total_base_us / total_v12_us, (blob.len()) as f64 / 1e9 / (total_v12_us * 1e-6));
        println!("==========================================================================================");
    }

    Ok(())
}
