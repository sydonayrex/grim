//! WI-8: Decode GEMM Kernel Bake-off.
//!
//! Benchmarks `grim_decode_gemm_f16` vs `rocBLAS` (`rocblas_gemm_ex`)
//! across served decode shapes: M in {1, 2, 4, 8} and typical projection dimensions (N, K).

use std::time::Instant;
use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::decode_test_shim;
use grim_tensor::backend::{BackendStorage, MemoryOps};
use grim_tensor::{DType, Shape};

const TEST_SHAPES: &[(usize, usize, usize, &str)] = &[
    // (M, N, K, description)
    // Batch 1 (single-sequence decode)
    (1, 4096, 4096, "Llama-3 8B Q/K/V/O"),
    (1, 14336, 4096, "Llama-3 8B Gate/Up"),
    (1, 4096, 14336, "Llama-3 8B Down"),
    (1, 8192, 8192, "70B Q/K/V/O"),
    (1, 28672, 8192, "70B Gate/Up"),
    (1, 8192, 28672, "70B Down"),

    // Batch 2
    (2, 4096, 4096, "M=2 Q/K/V/O"),
    (2, 14336, 4096, "M=2 Gate/Up"),
    (2, 4096, 14336, "M=2 Down"),

    // Batch 4
    (4, 4096, 4096, "M=4 Q/K/V/O"),
    (4, 14336, 4096, "M=4 Gate/Up"),
    (4, 4096, 14336, "M=4 Down"),

    // Batch 8
    (8, 4096, 4096, "M=8 Q/K/V/O"),
    (8, 14336, 4096, "M=8 Gate/Up"),
    (8, 4096, 14336, "M=8 Down"),
];

fn main() {
    let ordinal: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    const WARMUP: usize = 15;

    let dev = match RocmDevice::try_new(ordinal) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no ROCm device {ordinal}: {e:?}");
            std::process::exit(1);
        }
    };

    println!("# Decode GEMM Kernel Bake-off (Device: {}, Arch: {})", ordinal, dev.gcn_arch());
    println!("Warmup: {} iters, Timed: {} iters\n", WARMUP, iters);
    println!("| M | N | K | Shape Note | Decode (µs) | rocBLAS (µs) | WMMA-T R3 (µs) | WMMA-T R4 (µs) | rocBLAS BW | R3 BW | R4 BW | R3 Parity | R4 Parity |");
    println!("|---|---|---|------------|-------------|--------------|----------------|----------------|------------|-------|-------|-----------|-----------|");

    for &(m, n, k, desc) in TEST_SHAPES {
        let a_f32: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 * 0.05) - 0.3).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 * 0.05) - 0.4).collect();

        // Also build column-major B for transposed test: shape [N, K], element (col, row) = b_f32[row * N + col]
        let mut b_col_f32 = vec![0.0f32; n * k];
        for r in 0..k {
            for c in 0..n {
                b_col_f32[c * k + r] = b_f32[r * n + c];
            }
        }

        let a_f16_bytes: Vec<u8> = a_f32.iter().flat_map(|&f| half::f16::from_f32(f).to_le_bytes()).collect();
        let b_f16_bytes: Vec<u8> = b_f32.iter().flat_map(|&f| half::f16::from_f32(f).to_le_bytes()).collect();
        let b_col_f16_bytes: Vec<u8> = b_col_f32.iter().flat_map(|&f| half::f16::from_f32(f).to_le_bytes()).collect();

        let shape_a = Shape::from_slice(&[m, k]);
        let shape_b = Shape::from_slice(&[k, n]);
        let shape_b_col = Shape::from_slice(&[n, k]);
        let shape_c = Shape::from_slice(&[m, n]);

        let a_box = dev.from_cpu_bytes(&a_f16_bytes, &shape_a, DType::F16).expect("alloc a");
        let b_box = dev.from_cpu_bytes(&b_f16_bytes, &shape_b, DType::F16).expect("alloc b");
        let b_col_box = dev.from_cpu_bytes(&b_col_f16_bytes, &shape_b_col, DType::F16).expect("alloc b_col");
        let out_decode_box = dev.alloc_storage(&shape_c, DType::F16).expect("alloc out decode");
        let out_rocblas_box = dev.alloc_storage(&shape_c, DType::F16).expect("alloc out rocblas");
        let out_wmma_t_box = dev.alloc_storage(&shape_c, DType::F16).expect("alloc out wmma_t");
        let out_wmma_r4_box = dev.alloc_storage(&shape_c, DType::F16).expect("alloc out wmma_r4");

        let a = a_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let b = b_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let b_col = b_col_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let out_decode = out_decode_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let out_rocblas = out_rocblas_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let out_wmma_t = out_wmma_t_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let out_wmma_r4 = out_wmma_r4_box.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

        // 1. Warmup
        for _ in 0..WARMUP {
            decode_test_shim::launch_decode_gemm_f16(&dev, a, b, out_decode, m, n, k).unwrap();
            decode_test_shim::launch_rocblas_gemm_f16(&dev, a, b, out_rocblas, m, n, k).unwrap();
            decode_test_shim::launch_wmma_gemm_b_transposed(&dev, a, b_col, out_wmma_t, m, n, k).unwrap();
            decode_test_shim::launch_wmma_gemm_b_transposed_rdna4(&dev, a, b_col, out_wmma_r4, m, n, k).unwrap();
        }
        dev.synchronize();

        // Parity check vs rocBLAS
        let rocblas_res = out_rocblas.to_cpu_vec_f32().unwrap();
        let wmma_t_res = out_wmma_t.to_cpu_vec_f32().unwrap();
        let wmma_r4_res = out_wmma_r4.to_cpu_vec_f32().unwrap();

        let max_diff_wmma_t = wmma_t_res
            .iter()
            .zip(rocblas_res.iter())
            .map(|(w, r)| (w - r).abs())
            .fold(0.0f32, f32::max);

        let max_diff_wmma_r4 = wmma_r4_res
            .iter()
            .zip(rocblas_res.iter())
            .map(|(w, r)| (w - r).abs())
            .fold(0.0f32, f32::max);

        // 2. Benchmark decode_gemm
        let t0 = Instant::now();
        for _ in 0..iters {
            decode_test_shim::launch_decode_gemm_f16(&dev, a, b, out_decode, m, n, k).unwrap();
        }
        dev.synchronize();
        let decode_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // 3. Benchmark rocBLAS
        let t0 = Instant::now();
        for _ in 0..iters {
            decode_test_shim::launch_rocblas_gemm_f16(&dev, a, b, out_rocblas, m, n, k).unwrap();
        }
        dev.synchronize();
        let rocblas_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // 4. Benchmark WMMA-T (RDNA3 single-wave 16x32)
        let t0 = Instant::now();
        for _ in 0..iters {
            decode_test_shim::launch_wmma_gemm_b_transposed(&dev, a, b_col, out_wmma_t, m, n, k).unwrap();
        }
        dev.synchronize();
        let wmma_t_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // 5. Benchmark WMMA-T (RDNA4 multi-wave 16x64)
        let t0 = Instant::now();
        for _ in 0..iters {
            decode_test_shim::launch_wmma_gemm_b_transposed_rdna4(&dev, a, b_col, out_wmma_r4, m, n, k).unwrap();
        }
        dev.synchronize();
        let wmma_r4_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

        let total_bytes = ((m * k + k * n + m * n) * 2) as f64;
        let rocblas_bw = (total_bytes / (rocblas_us * 1e-6)) / 1e9;
        let wmma_t_bw = (total_bytes / (wmma_t_us * 1e-6)) / 1e9;
        let wmma_r4_bw = (total_bytes / (wmma_r4_us * 1e-6)) / 1e9;

        println!(
            "| {} | {:>5} | {:>5} | {:<18} | {:>10.1} | {:>10.1} | {:>14.1} | {:>14.1} | {:>10.1} | {:>5.1} | {:>5.1} | {:>9.2e} | {:>9.2e} |",
            m, n, k, desc, decode_us, rocblas_us, wmma_t_us, wmma_r4_us, rocblas_bw, wmma_t_bw, wmma_r4_bw, max_diff_wmma_t, max_diff_wmma_r4
        );
    }
}
