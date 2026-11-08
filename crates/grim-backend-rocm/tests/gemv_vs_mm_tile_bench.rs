//! Microbenchmark comparing baseline GEMV vs New MM-Tiled GEMM Variant:
//!   - Path A: Baseline FP32 GEMV (`launch_fp32_gemv_v1_for_ab` with `grim_fp32_gemv`)
//!   - Path B: Optimized GEMV v2 (`launch_fp32_gemv_v2_for_ab` with `grim_fp32_gemv_v2`)
//!   - Path C: New 2D MM-Tiled GEMV/GEMM (`launch_fp32_mm_tile_for_ab` with `grim_fp32_mm_tile`)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test gemv_vs_mm_tile_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::{gpu_test_enabled, RocmDevice, RocmStorage};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

fn upload_f32(dev: &RocmDevice, data: &[f32], shape: Shape) -> Box<dyn BackendStorage> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    dev.from_cpu_bytes(
        &bytes,
        &shape,
        DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        },
    )
    .expect("f32 upload")
}

struct BenchConfig {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

fn run_gemv_vs_mm_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;

    println!(
        "\n========================================================================================="
    );
    println!(
        "GEMV Baseline vs MM-Tile [{}]: M={}, N={}, K={}",
        cfg.name, m, n, k
    );
    println!(
        "========================================================================================="
    );

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let weight: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();

    let act_storage = upload_f32(&dev, &act, Shape::new(vec![m, k]));
    let weight_storage = upload_f32(&dev, &weight, Shape::new(vec![n, k]));

    let out_v1_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_v2_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_tile_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_splitk_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let weight_s = weight_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v1_s = out_v1_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v2_s = out_v2_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_tile_s = out_tile_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_splitk_s = out_splitk_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Parity Check
    print!("[1/5] Parity check v1 vs v2 vs MM-Tile vs SplitK-LDS... ");
    dev.launch_fp32_gemv_v1_for_ab(act_s, weight_s, out_v1_s, m, n, k).unwrap();
    dev.launch_fp32_gemv_v2_for_ab(act_s, weight_s, out_v2_s, m, n, k).unwrap();
    dev.launch_fp32_mm_tile_for_ab(act_s, weight_s, out_tile_s, m, n, k).unwrap();
    dev.launch_fp32_mm_splitk_lds_for_ab(act_s, weight_s, out_splitk_s, m, n, k, 4).unwrap();
    dev.synchronize();

    let res_v1 = out_v1_s.to_cpu_vec_f32().unwrap();
    let res_v2 = out_v2_s.to_cpu_vec_f32().unwrap();
    let res_tile = out_tile_s.to_cpu_vec_f32().unwrap();
    let res_splitk = out_splitk_s.to_cpu_vec_f32().unwrap();

    let mut max_diff_v2 = 0.0f32;
    let mut max_diff_tile = 0.0f32;
    let mut max_diff_splitk = 0.0f32;
    for i in 0..res_v1.len() {
        let d2 = (res_v1[i] - res_v2[i]).abs();
        if d2 > max_diff_v2 { max_diff_v2 = d2; }
        let dt = (res_v1[i] - res_tile[i]).abs();
        if dt > max_diff_tile { max_diff_tile = dt; }
        let dsk = (res_v1[i] - res_splitk[i]).abs();
        if dsk > max_diff_splitk { max_diff_splitk = dsk; }
    }
    println!("OK (diff v2={:.2e}, diff MM-Tile={:.2e}, diff SplitK-LDS={:.2e})", max_diff_v2, max_diff_tile, max_diff_splitk);
    assert!(max_diff_tile < 1e-4, "Parity check failed for MM-Tile");
    assert!(max_diff_splitk < 1e-4, "Parity check failed for SplitK-LDS");

    // 2. Timing GEMV v1 (Baseline)
    for _ in 0..cfg.warmup_iters {
        dev.launch_fp32_gemv_v1_for_ab(act_s, weight_s, out_v1_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/5] Timing Baseline GEMV v1... ");
    let mut times_v1: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_fp32_gemv_v1_for_ab(act_s, weight_s, out_v1_s, m, n, k).unwrap();
        dev.synchronize();
        times_v1.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Timing GEMV v2 (Unrolled)
    for _ in 0..cfg.warmup_iters {
        dev.launch_fp32_gemv_v2_for_ab(act_s, weight_s, out_v2_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[3/5] Timing GEMV v2 (Unrolled)... ");
    let mut times_v2: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_fp32_gemv_v2_for_ab(act_s, weight_s, out_v2_s, m, n, k).unwrap();
        dev.synchronize();
        times_v2.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 4. Timing New MM-Tile
    for _ in 0..cfg.warmup_iters {
        dev.launch_fp32_mm_tile_for_ab(act_s, weight_s, out_tile_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[4/5] Timing New 2D MM-Tile... ");
    let mut times_tile: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_fp32_mm_tile_for_ab(act_s, weight_s, out_tile_s, m, n, k).unwrap();
        dev.synchronize();
        times_tile.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 5. Timing SplitK-LDS
    for _ in 0..cfg.warmup_iters {
        dev.launch_fp32_mm_splitk_lds_for_ab(act_s, weight_s, out_splitk_s, m, n, k, 4).unwrap();
    }
    dev.synchronize();

    print!("[5/5] Timing Split-K (x4) + LDS Activation Caching... ");
    let mut times_splitk: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_fp32_mm_splitk_lds_for_ab(act_s, weight_s, out_splitk_s, m, n, k, 4).unwrap();
        dev.synchronize();
        times_splitk.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_v1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_v2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_tile.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_splitk.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_v1 = times_v1[cfg.bench_iters / 2];
    let med_v2 = times_v2[cfg.bench_iters / 2];
    let med_tile = times_tile[cfg.bench_iters / 2];
    let med_splitk = times_splitk[cfg.bench_iters / 2];

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<30} | {:<16} | {:<16} | {:<16} | {:<16}", "Metric", "GEMV v1 (Base)", "GEMV v2", "MM-Tile", "SplitK-LDS (x4)");
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<30} | {:>13.2} us | {:>13.2} us | {:>13.2} us | {:>13.2} us",
             "Latency (Sync Median)", med_v1, med_v2, med_tile, med_splitk);
    println!("{:<30} | {:>16} | {:>13.2}x | {:>13.2}x | {:>13.2}x",
             "Speedup vs Baseline", "1.00x", med_v1 / med_v2, med_v1 / med_tile, med_v1 / med_splitk);
    println!("-------------------------------------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_gemv_vs_mm_tile() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Production Qwen Decode Projections:
    // Shape 1: Qwen Decode FFN Intermediate (M=1, N=17408, K=5120)
    run_gemv_vs_mm_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Decode (M=1)",
        m: 1,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 2: Qwen Decode FFN Down (M=1, N=5120, K=17408)
    run_gemv_vs_mm_bench(&BenchConfig {
        name: "Qwen FFN Down Decode (M=1)",
        m: 1,
        n: 5120,
        k: 17408,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 3: Batched / Speculative Decode (M=4, N=17408, K=5120)
    run_gemv_vs_mm_bench(&BenchConfig {
        name: "Qwen Batched Decode (M=4)",
        m: 4,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });
}
