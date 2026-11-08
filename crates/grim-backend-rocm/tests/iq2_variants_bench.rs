//! Parity and benchmark test for IQ2_XXS and IQ2_S dot4 GEMV vs stock/LDS fused dequant kernels.
//!
//! Verifies:
//! 1. Parity between dot4 GEMV (M=1, M=2) and fused dequant baseline / CPU oracle.
//! 2. Benchmarks performance on >= 60 MB tensors:
//!    - IQ2_XXS: N=32768, K=5120 (48640 * 20 * 66 = 64,204,800 bytes = 61.23 MB)
//!    - IQ2_S:   N=28672, K=5120 (39424 * 20 * 82 = 64,655,360 bytes = 61.66 MB)

use grim_backend_rocm::{as_rocm, RocmDevice};
use grim_tensor::{ArithType, BackendStorage, DType, KQuantScheme, MemoryOps, Shape, Storage};
use std::time::Instant;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
    }
}

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn f32_vec(t: &Box<dyn BackendStorage>) -> Result<Vec<f32>, String> {
    Ok(as_rocm(t.as_ref())
        .map_err(|e| e.to_string())?
        .copy_to_host()
        .map_err(|e| format!("copy_to_host: {e}"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[test]
fn test_iq2_parity_and_variants() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("==========================================================================================");
    println!("IQ2_XXS & IQ2_S PARITY VERIFICATION (K=5120, N=256)");
    println!("==========================================================================================");

    let k = 5120;
    let n_test = 256;

    // Test IQ2_XXS
    {
        println!("
Testing IQ2_XXS parity...");
        let mut rng = Lcg(0x12345678);
        let vals: Vec<f32> = (0..n_test * k).map(|_| rng.next_f32()).collect();
        let packed = grim_quant::quant_iq2xxs(&vals).map_err(|e| format!("quant: {e}"))?;
        let w_st = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n_test, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ2XXS),
            },
        )?;

        for m in [1, 2] {
            let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let a_st = MemoryOps::from_cpu_bytes(
                &dev,
                f32_bytes(&a_vals),
                &Shape::new(vec![m * k]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let q81_len = m * (k / 32) * 36;
            let q81_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![q81_len]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_quantize_q8_1(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(q81_st.as_ref()).unwrap(),
                m,
                k,
            )?;

            let out_ref_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;
            let out_dot4_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            // Launch fused baseline (stock)
            dev.launch_fused_dequant_gemm_iq2xxs_stock_for_ab(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_ref_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;

            // Launch dot4
            dev.launch_dot4_iq2xxs_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_dot4_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;
            dev.synchronize();

            let out_ref = f32_vec(&out_ref_st)?;
            let out_dot4 = f32_vec(&out_dot4_st)?;

            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f32;
            let mut max_rel = 0.0f32;
            for i in 0..m * n_test {
                let diff = (out_ref[i] - out_dot4[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                sum_diff += diff;
                let rel = diff / (out_ref[i].abs().max(1.0));
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            let avg_diff = sum_diff / (m * n_test) as f32;
            println!("  IQ2_XXS M={m} vs fused stock: max_diff={max_diff:.4}, avg_diff={avg_diff:.6}, max_rel={max_rel:.4}");

            // Parity check v2_lds matches base dot4 exactly
            let out_v2_lds_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_dot4_iq2xxs_q81_gemv_v2_lds_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_v2_lds_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;
            dev.synchronize();
            let out_v2_lds = f32_vec(&out_v2_lds_st)?;
            let mut max_diff_lds = 0.0f32;
            for i in 0..m * n_test {
                let diff = (out_dot4[i] - out_v2_lds[i]).abs();
                if diff > max_diff_lds {
                    max_diff_lds = diff;
                }
            }
            println!("  IQ2_XXS M={m} v2_lds vs dot4: max_diff={max_diff_lds:.6}");
            assert!(max_diff_lds < 1e-4, "IQ2_XXS v2_lds matches base dot4");

            if m == 2 {
                let out_r2_st = MemoryOps::alloc_storage(
                    &dev,
                    &Shape::new(vec![m * n_test]),
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )?;
                dev.launch_dot4_iq2xxs_q81_gemv_rows_r2_for_ab(
                    as_rocm(q81_st.as_ref()).unwrap(),
                    as_rocm(w_st.as_ref()).unwrap(),
                    as_rocm(out_r2_st.as_ref()).unwrap(),
                    m,
                    n_test,
                    k,
                )?;
                dev.synchronize();

                let out_r2 = f32_vec(&out_r2_st)?;
                let mut max_diff_r2 = 0.0f32;
                for i in 0..m * n_test {
                    let diff = (out_dot4[i] - out_r2[i]).abs();
                    if diff > max_diff_r2 {
                        max_diff_r2 = diff;
                    }
                }
                println!("  IQ2_XXS M=2 rows_r2 vs dot4: max_diff={max_diff_r2:.6}");
                assert!(max_diff_r2 < 1e-4, "IQ2_XXS rows_r2 matches dot4");
            }
        }
    }

    
    // Test IQ2_XS
    {
        println!("
Testing IQ2_XS parity...");
        let mut rng = Lcg(0x56789ABC);
        let vals: Vec<f32> = (0..n_test * k).map(|_| rng.next_f32()).collect();
        let packed = grim_quant::quant_iq2xs(&vals).map_err(|e| format!("quant: {e}"))?;
        let w_st = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n_test, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ2XS),
            },
        )?;

        for m in [1, 2] {
            let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let a_st = MemoryOps::from_cpu_bytes(
                &dev,
                f32_bytes(&a_vals),
                &Shape::new(vec![m * k]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let q81_len = m * (k / 32) * 36;
            let q81_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![q81_len]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_quantize_q8_1(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(q81_st.as_ref()).unwrap(),
                m,
                k,
            )?;

            let out_ref_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;
            let out_dot4_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            dev.launch_fused_dequant_gemm_iq2xs_stock_for_ab(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_ref_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;

            dev.launch_dot4_iq2xs_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_dot4_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;
            dev.synchronize();

            let out_ref = f32_vec(&out_ref_st)?;
            let out_dot4 = f32_vec(&out_dot4_st)?;

            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f32;
            let mut max_rel = 0.0f32;
            for i in 0..m * n_test {
                let diff = (out_ref[i] - out_dot4[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                sum_diff += diff;
                let scale = out_ref[i].abs().max(out_dot4[i].abs()).max(1e-4);
                let rel = diff / scale;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            println!(
                "  IQ2_XS M={m} vs fused stock: max_diff={max_diff:.4}, avg_diff={:.6}, max_rel={max_rel:.4}",
                sum_diff / (m * n_test) as f32
            );

            // Test v2_lds
            let out_v2_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_dot4_iq2xs_q81_gemv_v2_lds_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_v2_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;
            dev.synchronize();
            let out_v2 = f32_vec(&out_v2_st)?;
            let mut max_diff_v2 = 0.0f32;
            for i in 0..m * n_test {
                let diff = (out_dot4[i] - out_v2[i]).abs();
                if diff > max_diff_v2 {
                    max_diff_v2 = diff;
                }
            }
            println!("  IQ2_XS M={m} v2_lds vs dot4: max_diff={max_diff_v2:.6}");
            assert!(max_diff_v2 < 1e-4, "IQ2_XS v2_lds mismatch vs base dot4");

            if m == 2 {
                let out_r2_st = MemoryOps::alloc_storage(
                    &dev,
                    &Shape::new(vec![m * n_test]),
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )?;
                dev.launch_dot4_iq2xs_q81_gemv_rows_r2_for_ab(
                    as_rocm(q81_st.as_ref()).unwrap(),
                    as_rocm(w_st.as_ref()).unwrap(),
                    as_rocm(out_r2_st.as_ref()).unwrap(),
                    m,
                    n_test,
                    k,
                )?;
                dev.synchronize();
                let out_r2 = f32_vec(&out_r2_st)?;
                let mut max_diff_r2 = 0.0f32;
                for i in 0..m * n_test {
                    let diff = (out_dot4[i] - out_r2[i]).abs();
                    if diff > max_diff_r2 {
                        max_diff_r2 = diff;
                    }
                }
                println!("  IQ2_XS M=2 rows_r2 vs dot4: max_diff={max_diff_r2:.6}");
                assert!(max_diff_r2 < 1e-4, "IQ2_XS rows_r2 mismatch vs base dot4");
            }
        }
    }

// Test IQ2_S
    {
        println!("
Testing IQ2_S parity...");
        let mut rng = Lcg(0x87654321);
        let vals: Vec<f32> = (0..n_test * k).map(|_| rng.next_f32()).collect();
        let packed = grim_quant::quant_iq2s(&vals).map_err(|e| format!("quant: {e}"))?;
        let w_st = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n_test, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ2S),
            },
        )?;

        for m in [1, 2] {
            let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let a_st = MemoryOps::from_cpu_bytes(
                &dev,
                f32_bytes(&a_vals),
                &Shape::new(vec![m * k]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let q81_len = m * (k / 32) * 36;
            let q81_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![q81_len]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_quantize_q8_1(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(q81_st.as_ref()).unwrap(),
                m,
                k,
            )?;

            let out_ref_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;
            let out_dot4_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            // Launch fused baseline (stock)
            dev.launch_fused_dequant_gemm_iq2s_stock_for_ab(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_ref_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;

            // Launch dot4
            dev.launch_dot4_iq2s_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_dot4_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;
            dev.synchronize();

            let out_ref = f32_vec(&out_ref_st)?;
            let out_dot4 = f32_vec(&out_dot4_st)?;

            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f32;
            let mut max_rel = 0.0f32;
            for i in 0..m * n_test {
                let diff = (out_ref[i] - out_dot4[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                sum_diff += diff;
                let rel = diff / (out_ref[i].abs().max(1.0));
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            let avg_diff = sum_diff / (m * n_test) as f32;
            println!("  IQ2_S M={m} vs fused stock: max_diff={max_diff:.4}, avg_diff={avg_diff:.6}, max_rel={max_rel:.4}");

            // Parity check v2_lds matches base dot4 exactly
            let out_v2_lds_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n_test]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_dot4_iq2s_q81_gemv_v2_lds_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                as_rocm(w_st.as_ref()).unwrap(),
                as_rocm(out_v2_lds_st.as_ref()).unwrap(),
                m,
                n_test,
                k,
            )?;
            dev.synchronize();
            let out_v2_lds = f32_vec(&out_v2_lds_st)?;
            let mut max_diff_lds = 0.0f32;
            for i in 0..m * n_test {
                let diff = (out_dot4[i] - out_v2_lds[i]).abs();
                if diff > max_diff_lds {
                    max_diff_lds = diff;
                }
            }
            println!("  IQ2_S M={m} v2_lds vs dot4: max_diff={max_diff_lds:.6}");
            assert!(max_diff_lds < 1e-4, "IQ2_S v2_lds matches base dot4");

            if m == 2 {
                let out_r2_st = MemoryOps::alloc_storage(
                    &dev,
                    &Shape::new(vec![m * n_test]),
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )?;
                dev.launch_dot4_iq2s_q81_gemv_rows_r2_for_ab(
                    as_rocm(q81_st.as_ref()).unwrap(),
                    as_rocm(w_st.as_ref()).unwrap(),
                    as_rocm(out_r2_st.as_ref()).unwrap(),
                    m,
                    n_test,
                    k,
                )?;
                dev.synchronize();

                let out_r2 = f32_vec(&out_r2_st)?;
                let mut max_diff_r2 = 0.0f32;
                for i in 0..m * n_test {
                    let diff = (out_dot4[i] - out_r2[i]).abs();
                    if diff > max_diff_r2 {
                        max_diff_r2 = diff;
                    }
                }
                println!("  IQ2_S M=2 rows_r2 vs dot4: max_diff={max_diff_r2:.6}");
                assert!(max_diff_r2 < 1e-4, "IQ2_S rows_r2 matches dot4");
            }
        }
    }

    Ok(())
}

#[test]
fn bench_iq2_all_variants_60mb() -> TestResult {
    let Some(dev) = gpu_device() else {
        println!("GPU tests disabled.");
        return Ok(());
    };

    println!("
==========================================================================================");
    println!("IQ2_XXS & IQ2_S 60 MB BENCHMARK ON RX 9070 XT (K=5120)");
    println!("==========================================================================================");

    let k = 5120;
    let iters = 25;

    // --- Bench IQ2_XXS ---
    // N=32768, K=5120 -> 48640 * 20 * 66 = 64,204,800 bytes = 61.23 MB
    {
        let n = 48640;
        let tensor_bytes = n * (k / 256) * 66;
        println!("
--- IQ2_XXS: N={n}, K={k} ({:.2} MB weight tensor) ---", tensor_bytes as f64 / 1e6);

        // Pre-quantize 1 column (20 superblocks = 1960 bytes) and tile across all N columns
        // to avoid waiting minutes on CPU quantization of 167M weights.
        let mut rng = Lcg(0xFEED1);
        let col_vals: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let col_packed = grim_quant::quant_iq2xxs(&col_vals).map_err(|e| format!("quant: {e}"))?;
        assert_eq!(col_packed.len(), (k / 256) * 66);

        let mut packed = Vec::with_capacity(tensor_bytes);
        for _ in 0..n {
            packed.extend_from_slice(&col_packed);
        }

        let w_st = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ2XXS),
            },
        )?;

        for m in [1, 2] {
            let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let a_st = MemoryOps::from_cpu_bytes(
                &dev,
                f32_bytes(&a_vals),
                &Shape::new(vec![m * k]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let q81_len = m * (k / 32) * 36;
            let q81_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![q81_len]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_quantize_q8_1(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(q81_st.as_ref()).unwrap(),
                m,
                k,
            )?;

            let y_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let a_r = as_rocm(a_st.as_ref()).unwrap();
            let q81_r = as_rocm(q81_st.as_ref()).unwrap();
            let w_r = as_rocm(w_st.as_ref()).unwrap();
            let y_r = as_rocm(y_st.as_ref()).unwrap();

            // Timing Stock Fused
            for _ in 0..3 {
                dev.launch_fused_dequant_gemm_iq2xxs_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t0 = Instant::now();
            for _ in 0..iters {
                dev.launch_fused_dequant_gemm_iq2xxs_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_stock = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing LDS Fused
            for _ in 0..3 {
                dev.launch_fused_dequant_gemm_iq2xxs_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t1 = Instant::now();
            for _ in 0..iters {
                dev.launch_fused_dequant_gemm_iq2xxs_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_lds = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 GEMV
            for _ in 0..3 {
                dev.launch_dot4_iq2xxs_q81_gemv_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t2 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2xxs_q81_gemv_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_dot4 = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v2_lds
            for _ in 0..3 {
                dev.launch_dot4_iq2xxs_q81_gemv_v2_lds_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_lds = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2xxs_q81_gemv_v2_lds_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_v2_lds = t_lds.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v3_col8
            for _ in 0..3 {
                dev.launch_dot4_iq2xxs_q81_gemv_v3_col8_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_col8 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2xxs_q81_gemv_v3_col8_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_v3_col8 = t_col8.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v4_prefetch
            for _ in 0..3 {
                // dev.launch_dot4_iq2xxs_q81_gemv_v4_prefetch_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            // let t_pref = Instant::now();
            for _ in 0..iters {
                // dev.launch_dot4_iq2xxs_q81_gemv_v4_prefetch_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            // let dur_v4_pref = 0.0; // t_pref.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            let bw = |dur: f64| (tensor_bytes as f64 * m as f64) / (dur * 1e-6) / 1e9;

            println!("
Latency & Bandwidth for IQ2_XXS M={m}:");
            println!("  Fused Stock:                {dur_stock:6.2} us | eff BW: {:6.1} GB/s ", bw(dur_stock));
            println!("  Fused LDS:                  {dur_lds:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_lds), dur_stock / dur_lds);
            println!("  Dot4 GEMV (base):           {dur_dot4:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_dot4), dur_stock / dur_dot4);
            println!("  Dot4 v2_lds (LDS grid):     {dur_v2_lds:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v2_lds), dur_stock / dur_v2_lds);
            println!("  Dot4 v3_col8 (8-col tile):  {dur_v3_col8:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v3_col8), dur_stock / dur_v3_col8);
            // println!("  Dot4 v4_pref (pipelined):   {dur_v4_pref:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v4_pref), dur_stock / dur_v4_pref);

            if m == 2 {
                for _ in 0..3 {
                    dev.launch_dot4_iq2xxs_q81_gemv_rows_r2_for_ab(q81_r, w_r, y_r, m, n, k)?;
                }
                dev.synchronize();
                let t3 = Instant::now();
                for _ in 0..iters {
                    dev.launch_dot4_iq2xxs_q81_gemv_rows_r2_for_ab(q81_r, w_r, y_r, m, n, k)?;
                }
                dev.synchronize();
                let dur_r2 = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

                println!(
                    "  Dot4 rows_r2 (dual-row):    {dur_r2:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x total, {:6.2} us/tok)",
                    bw(dur_r2),
                    dur_stock / dur_r2,
                    dur_r2 / 2.0
                );
            }
        }
    }

    // --- Bench IQ2_S ---
    // N=28672, K=5120 -> 39424 * 20 * 82 = 64,655,360 bytes = 61.66 MB
    {
        let n = 39424;
        let tensor_bytes = n * (k / 256) * 82;
        println!("
--- IQ2_S: N={n}, K={k} ({:.2} MB weight tensor) ---", tensor_bytes as f64 / 1e6);

        // Pre-quantize 1 column (20 superblocks = 2200 bytes) and tile across all N columns
        let mut rng = Lcg(0xFEED2);
        let col_vals: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let col_packed = grim_quant::quant_iq2s(&col_vals).map_err(|e| format!("quant: {e}"))?;
        assert_eq!(col_packed.len(), (k / 256) * 82);

        let mut packed = Vec::with_capacity(tensor_bytes);
        for _ in 0..n {
            packed.extend_from_slice(&col_packed);
        }

        let w_st = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ2S),
            },
        )?;

        for m in [1, 2] {
            let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let a_st = MemoryOps::from_cpu_bytes(
                &dev,
                f32_bytes(&a_vals),
                &Shape::new(vec![m * k]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let q81_len = m * (k / 32) * 36;
            let q81_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![q81_len]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_quantize_q8_1(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(q81_st.as_ref()).unwrap(),
                m,
                k,
            )?;

            let y_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let a_r = as_rocm(a_st.as_ref()).unwrap();
            let q81_r = as_rocm(q81_st.as_ref()).unwrap();
            let w_r = as_rocm(w_st.as_ref()).unwrap();
            let y_r = as_rocm(y_st.as_ref()).unwrap();

            // Timing Stock Fused
            for _ in 0..3 {
                dev.launch_fused_dequant_gemm_iq2s_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t0 = Instant::now();
            for _ in 0..iters {
                dev.launch_fused_dequant_gemm_iq2s_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_stock = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing LDS Fused
            for _ in 0..3 {
                dev.launch_fused_dequant_gemm_iq2s_lds_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t1 = Instant::now();
            for _ in 0..iters {
                dev.launch_fused_dequant_gemm_iq2s_lds_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_lds = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 GEMV
            for _ in 0..3 {
                dev.launch_dot4_iq2s_q81_gemv_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t2 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2s_q81_gemv_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_dot4 = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v2_lds
            for _ in 0..3 {
                dev.launch_dot4_iq2s_q81_gemv_v2_lds_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_lds = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2s_q81_gemv_v2_lds_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_v2_lds = t_lds.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v3_col8
            for _ in 0..3 {
                dev.launch_dot4_iq2s_q81_gemv_v3_col8_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_col8 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2s_q81_gemv_v3_col8_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_v3_col8 = t_col8.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v4_prefetch
            for _ in 0..3 {
                // dev.launch_dot4_iq2s_q81_gemv_v4_prefetch_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            // let t_pref = Instant::now();
            for _ in 0..iters {
                // dev.launch_dot4_iq2s_q81_gemv_v4_prefetch_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            // let dur_v4_pref = 0.0; // t_pref.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            let bw = |dur: f64| (tensor_bytes as f64 * m as f64) / (dur * 1e-6) / 1e9;

            println!("
Latency & Bandwidth for IQ2_S M={m}:");
            println!("  Fused Stock:                {dur_stock:6.2} us | eff BW: {:6.1} GB/s ", bw(dur_stock));
            println!("  Fused LDS:                  {dur_lds:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_lds), dur_stock / dur_lds);
            println!("  Dot4 GEMV (base):           {dur_dot4:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_dot4), dur_stock / dur_dot4);
            println!("  Dot4 v2_lds (LDS grid):     {dur_v2_lds:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v2_lds), dur_stock / dur_v2_lds);
            println!("  Dot4 v3_col8 (8-col tile):  {dur_v3_col8:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v3_col8), dur_stock / dur_v3_col8);
            // println!("  Dot4 v4_pref (pipelined):   {dur_v4_pref:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v4_pref), dur_stock / dur_v4_pref);

            if m == 2 {
                for _ in 0..3 {
                    dev.launch_dot4_iq2s_q81_gemv_rows_r2_for_ab(q81_r, w_r, y_r, m, n, k)?;
                }
                dev.synchronize();
                let t3 = Instant::now();
                for _ in 0..iters {
                    dev.launch_dot4_iq2s_q81_gemv_rows_r2_for_ab(q81_r, w_r, y_r, m, n, k)?;
                }
                dev.synchronize();
                let dur_r2 = t3.elapsed().as_secs_f64() * 1e6 / (iters as f64);

                println!(
                    "  Dot4 rows_r2 (dual-row):    {dur_r2:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x total, {:6.2} us/tok)",
                    bw(dur_r2),
                    dur_stock / dur_r2,
                    dur_r2 / 2.0
                );
            }
        }
    }

    // =========================================================================
    // IQ2_XS Benchmark
    // =========================================================================
    {
        let n = 43520;
        let tensor_bytes = n * (k / 256) * 74;
        println!("--- IQ2_XS: N={}, K={} ({:.2} MB weight tensor) ---", n, k, tensor_bytes as f64 / 1e6);

        let mut rng = Lcg(0xFEED3);
        let col_vals: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let col_packed = grim_quant::quant_iq2xs(&col_vals).map_err(|e| format!("quant: {e}"))?;
        assert_eq!(col_packed.len(), (k / 256) * 74);

        let mut packed = Vec::with_capacity(tensor_bytes);
        for _ in 0..n {
            packed.extend_from_slice(&col_packed);
        }

        let w_st = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ2XS),
            },
        )?;

        for m in [1, 2] {
            let a_vals: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let a_st = MemoryOps::from_cpu_bytes(
                &dev,
                f32_bytes(&a_vals),
                &Shape::new(vec![m * k]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let q81_len = m * (k / 32) * 36;
            let q81_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![q81_len]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )?;
            dev.launch_quantize_q8_1(
                as_rocm(a_st.as_ref()).unwrap(),
                as_rocm(q81_st.as_ref()).unwrap(),
                m,
                k,
            )?;

            let y_st = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![m * n]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )?;

            let a_r = as_rocm(a_st.as_ref()).unwrap();
            let q81_r = as_rocm(q81_st.as_ref()).unwrap();
            let w_r = as_rocm(w_st.as_ref()).unwrap();
            let y_r = as_rocm(y_st.as_ref()).unwrap();

            // Timing Stock Fused
            for _ in 0..3 {
                dev.launch_fused_dequant_gemm_iq2xs_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_stock = Instant::now();
            for _ in 0..iters {
                dev.launch_fused_dequant_gemm_iq2xs_stock_for_ab(a_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_stock = t_stock.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 Base
            for _ in 0..3 {
                dev.launch_dot4_iq2xs_q81_gemv_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_dot4 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2xs_q81_gemv_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_dot4 = t_dot4.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v2_lds
            for _ in 0..3 {
                dev.launch_dot4_iq2xs_q81_gemv_v2_lds_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_v2_lds = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2xs_q81_gemv_v2_lds_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_v2_lds = t_v2_lds.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            // Timing Dot4 v3_col8
            for _ in 0..3 {
                dev.launch_dot4_iq2xs_q81_gemv_v3_col8_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let t_col8 = Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq2xs_q81_gemv_v3_col8_for_ab(q81_r, w_r, y_r, m, n, k)?;
            }
            dev.synchronize();
            let dur_v3_col8 = t_col8.elapsed().as_secs_f64() * 1e6 / (iters as f64);

            let bw = |dur: f64| (tensor_bytes as f64 * m as f64) / (dur * 1e-6) / 1e9;

            println!("Latency & Bandwidth for IQ2_XS M={}:", m);
            println!("  Fused Stock:                {dur_stock:6.2} us | eff BW: {:6.1} GB/s ", bw(dur_stock));
            println!("  Dot4 GEMV (base):           {dur_dot4:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_dot4), dur_stock / dur_dot4);
            println!("  Dot4 v2_lds (LDS grid):     {dur_v2_lds:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v2_lds), dur_stock / dur_v2_lds);
            println!("  Dot4 v3_col8 (8-col tile):  {dur_v3_col8:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x)", bw(dur_v3_col8), dur_stock / dur_v3_col8);

            if m == 2 {
                for _ in 0..3 {
                    dev.launch_dot4_iq2xs_q81_gemv_rows_r2_for_ab(q81_r, w_r, y_r, m, n, k)?;
                }
                dev.synchronize();
                let t_r2 = Instant::now();
                for _ in 0..iters {
                    dev.launch_dot4_iq2xs_q81_gemv_rows_r2_for_ab(q81_r, w_r, y_r, m, n, k)?;
                }
                dev.synchronize();
                let dur_r2 = t_r2.elapsed().as_secs_f64() * 1e6 / (iters as f64);
                println!(
                    "  Dot4 rows_r2 (dual-row):    {dur_r2:6.2} us | eff BW: {:6.1} GB/s (speedup vs stock: {:4.2}x total, {:6.2} us/tok)",
                    bw(dur_r2),
                    dur_stock / dur_r2,
                    dur_r2 / 2.0
                );
            }
        }
    }

    Ok(())
}
    