//! Microbenchmark comparing:
//!   - Baseline: `grim_f16_mm_tile` (Normal)
//!   - Variant 1: `grim_f16_mm_tile_2row` (Multi-row weight reuse)
//!   - Variant 3: `grim_f16_mm_tile_prefetch` (Software pipelining / prefetching)
//!   - Variant 1+3: `grim_f16_mm_tile_2row_prefetch` (Multi-row + Prefetching)
//!   - Matrix Core WMMA: `launch_wmma_gemm_b_transposed`
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test f16_opt_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::{gpu_test_enabled, RocmDevice, RocmStorage};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

fn upload_f16(dev: &RocmDevice, data: &[f32], shape: Shape) -> Box<dyn BackendStorage> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&v| {
            let h = half::f16::from_f32(v);
            h.to_bits().to_ne_bytes()
        })
        .collect();
    dev.from_cpu_bytes(
        &bytes,
        &shape,
        DType {
            arith: ArithType::F16,
            storage: DTypeStorage::Native,
        },
    )
    .expect("f16 upload")
}

struct BenchConfig {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

fn run_f16_opt_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;

    println!(
        "\n========================================================================================="
    );
    println!(
        "FP16 MM Optimization Bench [{}]: M={}, N={}, K={}",
        cfg.name, m, n, k
    );
    println!(
        "========================================================================================="
    );

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let weight: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();

    let act_storage = upload_f16(&dev, &act, Shape::new(vec![m, k]));
    let weight_storage = upload_f16(&dev, &weight, Shape::new(vec![n, k]));

    let out_norm_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_v1_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_v3_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_v13_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_wmma_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let weight_s = weight_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_norm_s = out_norm_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v1_s = out_v1_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v3_s = out_v3_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v13_s = out_v13_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_wmma_s = out_wmma_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Correctness & Parity check
    print!("[1/6] Parity check Normal vs V1 vs V3 vs V1+3... ");
    dev.launch_f16_mm_tile_for_ab(act_s, weight_s, out_norm_s, m, n, k).unwrap();
    dev.launch_f16_mm_tile_2row_for_ab(act_s, weight_s, out_v1_s, m, n, k).unwrap();
    dev.launch_f16_mm_tile_prefetch_for_ab(act_s, weight_s, out_v3_s, m, n, k).unwrap();
    dev.launch_f16_mm_tile_2row_prefetch_for_ab(act_s, weight_s, out_v13_s, m, n, k).unwrap();
    dev.launch_wmma_gemm_b_transposed(act_s, weight_s, out_wmma_s, m, n, k).unwrap();
    dev.synchronize();

    let norm_res = out_norm_s.to_cpu_vec_f32().unwrap();
    let v1_res = out_v1_s.to_cpu_vec_f32().unwrap();
    let v3_res = out_v3_s.to_cpu_vec_f32().unwrap();
    let v13_res = out_v13_s.to_cpu_vec_f32().unwrap();

    let mut max_diff_v1 = 0.0f32;
    let mut max_diff_v3 = 0.0f32;
    let mut max_diff_v13 = 0.0f32;
    for i in 0..norm_res.len() {
        let d1 = (norm_res[i] - v1_res[i]).abs();
        if d1 > max_diff_v1 { max_diff_v1 = d1; }
        let d3 = (norm_res[i] - v3_res[i]).abs();
        if d3 > max_diff_v3 { max_diff_v3 = d3; }
        let d13 = (norm_res[i] - v13_res[i]).abs();
        if d13 > max_diff_v13 { max_diff_v13 = d13; }
    }
    println!("OK (diff V1={:.2e}, V3={:.2e}, V1+3={:.2e})", max_diff_v1, max_diff_v3, max_diff_v13);
    assert!(max_diff_v1 < 1e-3, "Parity failed for Variant 1 (2-Row)");
    assert!(max_diff_v3 < 1e-3, "Parity failed for Variant 3 (Prefetch)");
    assert!(max_diff_v13 < 1e-3, "Parity failed for Variant 1+3 (2-Row Prefetch)");

    // Helper closure to measure median latency
    let time_kernel = |label: &str, mut launch_fn: Box<dyn FnMut()>| -> f64 {
        for _ in 0..cfg.warmup_iters {
            launch_fn();
        }
        dev.synchronize();

        print!("Timing {}... ", label);
        let mut times: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
        for _ in 0..cfg.bench_iters {
            let t0 = Instant::now();
            launch_fn();
            dev.synchronize();
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        println!("DONE");
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[cfg.bench_iters / 2]
    };

    let med_norm = time_kernel("Normal MM-Tile", Box::new(|| {
        dev.launch_f16_mm_tile_for_ab(act_s, weight_s, out_norm_s, m, n, k).unwrap();
    }));

    let med_v1 = time_kernel("Variant 1 (2-Row)", Box::new(|| {
        dev.launch_f16_mm_tile_2row_for_ab(act_s, weight_s, out_v1_s, m, n, k).unwrap();
    }));

    let med_v3 = time_kernel("Variant 3 (Prefetch)", Box::new(|| {
        dev.launch_f16_mm_tile_prefetch_for_ab(act_s, weight_s, out_v3_s, m, n, k).unwrap();
    }));

    let med_v13 = time_kernel("Variant 1+3 (2-Row Prefetch)", Box::new(|| {
        dev.launch_f16_mm_tile_2row_prefetch_for_ab(act_s, weight_s, out_v13_s, m, n, k).unwrap();
    }));

    let med_wmma = time_kernel("Matrix Core WMMA", Box::new(|| {
        dev.launch_wmma_gemm_b_transposed(act_s, weight_s, out_wmma_s, m, n, k).unwrap();
    }));

    let total_bytes = (n * k * 2 + m * k * 2 + m * n * 2) as f64;
    let bw = |t: f64| (total_bytes / (t * 1e-6)) / 1e9;

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("--------------------------------------------------------------------------------------------------------------------------------");
    println!("{:<24} | {:<14} | {:<14} | {:<14} | {:<14} | {:<14}", "Kernel", "Latency (us)", "Bandwidth", "vs Normal", "vs WMMA", "Per-Token Time");
    println!("--------------------------------------------------------------------------------------------------------------------------------");
    println!("{:<24} | {:>11.2} us | {:>11.2} GB/s| {:>13.2}x | {:>13.2}x | {:>11.2} us",
             "Normal (Baseline)", med_norm, bw(med_norm), 1.0, med_wmma / med_norm, med_norm / (m as f64));
    println!("{:<24} | {:>11.2} us | {:>11.2} GB/s| {:>13.2}x | {:>13.2}x | {:>11.2} us",
             "Variant 1 (2-Row)", med_v1, bw(med_v1), med_norm / med_v1, med_wmma / med_v1, med_v1 / (m as f64));
    println!("{:<24} | {:>11.2} us | {:>11.2} GB/s| {:>13.2}x | {:>13.2}x | {:>11.2} us",
             "Variant 3 (Prefetch)", med_v3, bw(med_v3), med_norm / med_v3, med_wmma / med_v3, med_v3 / (m as f64));
    println!("{:<24} | {:>11.2} us | {:>11.2} GB/s| {:>13.2}x | {:>13.2}x | {:>11.2} us",
             "Variant 1+3 (Combined)", med_v13, bw(med_v13), med_norm / med_v13, med_wmma / med_v13, med_v13 / (m as f64));
    println!("{:<24} | {:>11.2} us | {:>11.2} GB/s| {:>13.2}x | {:>13.2}x | {:>11.2} us",
             "Matrix Core WMMA", med_wmma, bw(med_wmma), med_norm / med_wmma, 1.0, med_wmma / (m as f64));
    println!("--------------------------------------------------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_fp16_optimizations() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Shape 1: Decode (M=1, N=17408, K=5120)
    run_f16_opt_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Single Decode (M=1)",
        m: 1,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 2: Small Batch (M=2, N=17408, K=5120)
    run_f16_opt_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Dual Decode (M=2)",
        m: 2,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 3: Batched (M=4, N=17408, K=5120)
    run_f16_opt_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Quad Decode (M=4)",
        m: 4,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });
}
