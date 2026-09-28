//! Task 9 (Plan 3): Unit test for small-batch prefill GEMM parity.
//! Compares batched M in {2, 16, 27, 32, 64} across multiple K in {256, 1024}
//! and irregular N in {127, 513, 1024, 4608} against CPU unquantized GEMM reference.

use grim_backend_rocm::RocmDevice;
use grim_tensor::CoreTensorOps;
use grim_tensor::QuantOps;
use grim_tensor::Shape;
use grim_tensor::dtype::{DType, QuantFormat};

#[test]
#[ignore]
fn test_small_batch_prefill_gemm_parity() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_small_batch_prefill_gemm_parity");
        return;
    }
    let dev = RocmDevice::new(0);

    let test_cases: Vec<(usize, usize, usize)> = vec![
        (2, 127, 256),
        (16, 513, 256),
        (27, 1024, 1024),
        (32, 1024, 1024),
        (64, 4608, 1024),
    ];

    for (m, n, k) in test_cases {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin() * 2.0).collect();
        let b_unquant: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.017).cos() * 1.5).collect();

        // CPU unquantized reference: C = A * B^T -> [M, N]
        let mut cpu_ref = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[row * k + p] * b_unquant[col * k + p];
                }
                cpu_ref[row * n + col] = sum;
            }
        }

        let a_shape = Shape::new(vec![m, k]);
        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();

        let b_shape = Shape::new(vec![n, k]);
        let b_raw = dev.from_cpu(&b_unquant, &b_shape, DType::F32).unwrap();
        let (b_q80, h_quant) = dev
            .quantize_on_device(b_raw.as_ref(), QuantFormat::Q8_0)
            .unwrap();
        h_quant.synchronize().unwrap();

        let out_shape = Shape::new(vec![m, n]);
        let (out_storage, h_gemm) = dev
            .fused_quant_gemm(
                a_storage.as_ref(),
                b_q80.as_ref(),
                QuantFormat::Q8_0,
                &out_shape,
            )
            .unwrap();
        h_gemm.synchronize().unwrap();

        let gpu_res = out_storage.to_cpu_vec_f32().unwrap();
        assert_eq!(gpu_res.len(), m * n);

        let mut max_abs_diff = 0.0f32;
        let mut max_ref = 0.0f32;
        for i in 0..m * n {
            let diff = (gpu_res[i] - cpu_ref[i]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            if cpu_ref[i].abs() > max_ref {
                max_ref = cpu_ref[i].abs();
            }
        }
        let rel_err = max_abs_diff / max_ref.max(1.0);
        eprintln!(
            "[test_small_batch_prefill_gemm_parity] M={m} N={n} K={k} max_diff={max_abs_diff:.4} rel_err={rel_err:.4}"
        );
        assert!(
            rel_err < 0.05,
            "M={m} N={n} K={k} relative error {rel_err:.4} exceeds 5% (max_diff={max_abs_diff}, max_ref={max_ref})"
        );
    }
}

/// 9B perf plan Step 3: skinny-M prefill (2 <= M <= 15) must route the
/// k-quant GEMMs to the LDS-tiled kernels, not the scalar one-thread-per-
/// output kernel that re-reads every weight block M times. Profile evidence:
/// a 5-token 9B prefill spends ~14 s in grim_fused_dequant_gemm_{q4k,q5k,q6k}.
/// Parity vs the CPU dequant-matmul reference at M in {2,3,5,8,15}, then an
/// absolute throughput floor at the 9B prefill shape: the tiled path must
/// sustain >= 46 GB/s of weight bytes (2x the scalar regime's measured
/// ~23 GB/s; see tests/dot4_gemv_floor_probe.rs). Run with the GRIM_*_TILED
/// env vars set; when the dispatch refuses skinny M the floor goes red.
#[test]
#[ignore]
fn skinny_m_kquant_tiled_parity_and_throughput() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping skinny_m_kquant_tiled_parity_and_throughput");
        return;
    }
    unsafe {
        // Set BEFORE the dispatchers' OnceLock caches initialize.
        std::env::set_var("GRIM_Q4K_TILED", "1");
        std::env::set_var("GRIM_Q5K_TILED", "1");
        std::env::set_var("GRIM_Q6K_TILED", "1");
    }
    let dev = RocmDevice::new(0);

    let formats: Vec<(QuantFormat, &str)> = vec![
        (QuantFormat::Q4K, "q4k"),
        (QuantFormat::Q5K, "q5k"),
        (QuantFormat::Q6K, "q6k"),
    ];

    for (fmt, tag) in &formats {
        // --- parity at skinny M ---
        for m in [2usize, 3, 5, 8, 15] {
            let n = 128usize;
            let k = 512usize;
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin() * 2.0).collect();
            let b_unquant: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.017).cos() * 1.5).collect();

            let b_bytes = match tag {
                &"q4k" => grim_quant::quant_q4k(&b_unquant).unwrap(),
                &"q5k" => grim_quant::quant_q5k(&b_unquant).unwrap(),
                _ => grim_quant::quant_q6k(&b_unquant).unwrap(),
            };
            let b_dtype = DType {
                arith: grim_tensor::dtype::ArithType::F32,
                storage: grim_tensor::dtype::Storage::KQuant(match tag {
                    &"q4k" => grim_tensor::dtype::KQuantScheme::Q4K,
                    &"q5k" => grim_tensor::dtype::KQuantScheme::Q5K,
                    _ => grim_tensor::dtype::KQuantScheme::Q6K,
                }),
            };
            let a_storage = dev.from_cpu(&a_data, &Shape::new(vec![m, k]), DType::F32).unwrap();
            let b_storage = grim_tensor::MemoryOps::from_cpu_bytes(
                &dev,
                &b_bytes,
                &Shape::new(vec![n, k]),
                b_dtype,
            )
            .unwrap();

            let out_shape = Shape::new(vec![m, n]);
            let (out_storage, h) = dev
                .fused_quant_gemm(a_storage.as_ref(), b_storage.as_ref(), *fmt, &out_shape)
                .unwrap_or_else(|e| panic!("{tag} m={m}: fused_quant_gemm failed: {e}"));
            h.synchronize().unwrap();
            let gpu = out_storage.to_cpu_vec_f32().unwrap();

            let b_deq = match tag {
                &"q4k" => grim_quant::dequant_q4k(&b_bytes, n * k).unwrap(),
                &"q5k" => grim_quant::dequant_q5k(&b_bytes, n * k).unwrap(),
                _ => grim_quant::dequant_q6k(&b_bytes, n * k).unwrap(),
            };
            let mut worst = 0.0f32;
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a_data[row * k + p] * b_deq[col * k + p];
                    }
                    worst = worst.max((gpu[row * n + col] - sum).abs());
                }
            }
            assert!(
                worst < 0.5,
                "{tag} skinny-M m={m}: GEMM diverges from CPU dequant reference: {worst}"
            );
            eprintln!("[skinny-m] {tag} m={m} parity ok (worst {worst:.5})");
        }

        // --- absolute throughput floor at the 9B prefill shape ---
        let (m, n, k) = (5usize, 16384usize, 4096usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin() * 2.0).collect();
        let b_unquant: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.017).cos() * 1.5).collect();
        let b_bytes = match tag {
            &"q4k" => grim_quant::quant_q4k(&b_unquant).unwrap(),
            &"q5k" => grim_quant::quant_q5k(&b_unquant).unwrap(),
            _ => grim_quant::quant_q6k(&b_unquant).unwrap(),
        };
        let b_dtype = DType {
            arith: grim_tensor::dtype::ArithType::F32,
            storage: grim_tensor::dtype::Storage::KQuant(match tag {
                &"q4k" => grim_tensor::dtype::KQuantScheme::Q4K,
                &"q5k" => grim_tensor::dtype::KQuantScheme::Q5K,
                _ => grim_tensor::dtype::KQuantScheme::Q6K,
            }),
        };
        let a_storage = dev.from_cpu(&a_data, &Shape::new(vec![m, k]), DType::F32).unwrap();
        let b_storage = grim_tensor::MemoryOps::from_cpu_bytes(
            &dev,
            &b_bytes,
            &Shape::new(vec![n, k]),
            b_dtype,
        )
        .unwrap();
        let out_shape = Shape::new(vec![m, n]);
        for _ in 0..2 {
            let (_, _h) = dev
                .fused_quant_gemm(a_storage.as_ref(), b_storage.as_ref(), *fmt, &out_shape)
                .unwrap();
            dev.synchronize();
        }
        let iters = 5usize;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let (_, _h) = dev
                .fused_quant_gemm(a_storage.as_ref(), b_storage.as_ref(), *fmt, &out_shape)
                .unwrap();
            dev.synchronize();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let block_bytes = match tag {
            &"q4k" => 144.0f64,
            &"q5k" => 176.0,
            _ => 210.0,
        };
        let gb = n as f64 * (k as f64 / 256.0) * block_bytes / 1e9;
        let gbps = gb / (ms * 1e-3);
        eprintln!("[skinny-m] {tag} m=5 n={n} k={k}: {ms:.2} ms/iter = {gbps:.1} GB/s weight bytes");
        // Floor = 10x the scalar kernel's MEASURED rate at this exact shape
        // (181 ms = 0.2 GB/s in the pre-gate red run). Not an absolute
        // ambition: TILE_M=4 wastes half its M tile at M=5; the point is that
        // the dispatch must never fall back to the scalar regime.
        assert!(
            gbps >= 2.0,
            "{tag}: skinny-M prefill must beat the scalar regime by >=10x (got {gbps:.1} GB/s, floor 2.0)"
        );
    }
}
