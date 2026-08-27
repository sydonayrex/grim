//! Micro-benchmarks for backend operator timing (§WI-E9).
//! Measures dequantization throughput and packed-GEMM ops across shapes.
//! Emits per-op timing lines suitable for pasting into docs/benchmarks tables.

use std::time::Instant;

use grim_quant::{dequant_q4k, dequant_q80, gemm_q8_0_packed, quant_q4k, quant_q80};

fn main() {
    println!("=== Grim Backend Operator Timing Benchmark (§WI-E9) ===");

    for &k in &[512usize, 2048, 4096, 8192] {
        let mut weights = vec![0.0f32; k];
        for (i, slot) in weights.iter_mut().enumerate() {
            *slot = ((i % 100) as f32 - 50.0) * 0.02;
        }

        // Q8_0 dequant timing
        let q8 = quant_q80(&weights).expect("quant q8_0");
        let iters = 1000;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dequant_q80(&q8, k).expect("dequant q8_0");
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let gb_per_sec = (k * 4 * iters) as f64 / elapsed / (1024.0 * 1024.0 * 1024.0);
        println!(
            "Q8_0 dequant [k={:<5}]: {:.3} ms/iter | {:.2} GB/s",
            k,
            (elapsed / iters as f64) * 1000.0,
            gb_per_sec
        );

        // Q4_K dequant timing
        let q4 = quant_q4k(&weights).expect("quant q4_k");
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dequant_q4k(&q4, k).expect("dequant q4_k");
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let gb_per_sec = (k * 4 * iters) as f64 / elapsed / (1024.0 * 1024.0 * 1024.0);
        println!(
            "Q4_K dequant [k={:<5}]: {:.3} ms/iter | {:.2} GB/s",
            k,
            (elapsed / iters as f64) * 1000.0,
            gb_per_sec
        );
    }

    // Packed Q8_0 GEMM timing (WI-E7 kernels): [m,k] x [n,k]^T
    println!("--- gemm_q8_0_packed ---");
    for &(m, n, k) in &[
        (1usize, 1024usize, 2048usize),
        (1, 4096, 4096),
        (8, 4096, 4096),
    ] {
        let mut a = vec![0.0f32; m * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i % 97) as f32 - 48.0) * 0.01;
        }
        // Build B as n rows of k weights, quantize row-wise to packed bytes.
        let mut b_bytes = Vec::new();
        for r in 0..n {
            let row: Vec<f32> = (0..k)
                .map(|c| ((r * 31 + c % 89) as f32 - 44.0) * 0.02)
                .collect();
            b_bytes.extend_from_slice(&grim_quant::quant_q80(&row).expect("quant b row"));
        }
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = gemm_q8_0_packed(&a, &b_bytes, m, n, k).expect("packed gemm");
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
        println!(
            "gemm_q8_0_packed [m={m}, n={n}, k={k}]: {:.3} ms/iter | {:.2} GFLOP/s",
            (elapsed / iters as f64) * 1000.0,
            flops / elapsed / 1e9
        );
    }

    println!("=== done ===");
}
