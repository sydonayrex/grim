//! Microbenchmark comparing FP16 GEMV Baseline vs FP16 MM-Tile:
//!   - Baseline: `grim_dot2_f16_gemv` (1 wave/4 cols, v_dot2_f32_f16)
//!   - MM-Tile: `grim_f16_mm_tile` (256 threads/64 cols, uint4 vector loads, v_dot2_f32_f16)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test f16_gemv_bench -- --nocapture

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

fn run_f16_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;

    println!(
        "\n========================================================================================="
    );
    println!(
        "FP16 GEMV Baseline vs FP16 MM-Tile [{}]: M={}, N={}, K={}",
        cfg.name, m, n, k
    );
    println!(
        "========================================================================================="
    );

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let weight: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();

    let act_storage = upload_f16(&dev, &act, Shape::new(vec![m, k]));
    let weight_storage = upload_f16(&dev, &weight, Shape::new(vec![n, k]));

    let out_base_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_tile_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_wmma_storage = upload_f16(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let weight_s = weight_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_base_s = out_base_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_tile_s = out_tile_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_wmma_s = out_wmma_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Parity Check
    print!("[1/4] Parity check Baseline vs MM-Tile vs Matrix Core WMMA... ");
    dev.launch_dot2_f16_gemv_for_ab(act_s, weight_s, out_base_s, m, n, k).unwrap();
    dev.launch_f16_mm_tile_for_ab(act_s, weight_s, out_tile_s, m, n, k).unwrap();
    dev.launch_wmma_gemm_b_transposed(act_s, weight_s, out_wmma_s, m, n, k).unwrap();
    dev.synchronize();

    let base_res = out_base_s.to_cpu_vec_f32().unwrap();
    let tile_res = out_tile_s.to_cpu_vec_f32().unwrap();
    let wmma_res = out_wmma_s.to_cpu_vec_f32().unwrap();

    let mut max_diff_tile = 0.0f32;
    let mut max_diff_wmma = 0.0f32;
    for i in 0..base_res.len() {
        let dt = (base_res[i] - tile_res[i]).abs();
        if dt > max_diff_tile { max_diff_tile = dt; }
        let dw = (base_res[i] - wmma_res[i]).abs();
        if dw > max_diff_wmma { max_diff_wmma = dw; }
    }
    println!("OK (diff tile={:.2e}, diff wmma={:.2e})", max_diff_tile, max_diff_wmma);
    assert!(max_diff_tile < 1e-3, "Parity check failed for FP16 MM-Tile");
    assert!(max_diff_wmma < 1e-2, "Parity check failed for FP16 WMMA");

    // 2. Timing Baseline
    for _ in 0..cfg.warmup_iters {
        dev.launch_dot2_f16_gemv_for_ab(act_s, weight_s, out_base_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/4] Timing FP16 GEMV Baseline... ");
    let mut times_base: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_dot2_f16_gemv_for_ab(act_s, weight_s, out_base_s, m, n, k).unwrap();
        dev.synchronize();
        times_base.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Timing MM-Tile
    for _ in 0..cfg.warmup_iters {
        dev.launch_f16_mm_tile_for_ab(act_s, weight_s, out_tile_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[3/4] Timing FP16 MM-Tile... ");
    let mut times_tile: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_f16_mm_tile_for_ab(act_s, weight_s, out_tile_s, m, n, k).unwrap();
        dev.synchronize();
        times_tile.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 4. Timing Matrix Core WMMA
    for _ in 0..cfg.warmup_iters {
        dev.launch_wmma_gemm_b_transposed(act_s, weight_s, out_wmma_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[4/4] Timing FP16 Matrix Core WMMA... ");
    let mut times_wmma: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_wmma_gemm_b_transposed(act_s, weight_s, out_wmma_s, m, n, k).unwrap();
        dev.synchronize();
        times_wmma.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_base.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_tile.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_wmma.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_base = times_base[cfg.bench_iters / 2];
    let med_tile = times_tile[cfg.bench_iters / 2];
    let med_wmma = times_wmma[cfg.bench_iters / 2];

    let total_bytes = (n * k * 2 + m * k * 2 + m * n * 2) as f64;
    let bw_base = (total_bytes / (med_base * 1e-6)) / 1e9;
    let bw_tile = (total_bytes / (med_tile * 1e-6)) / 1e9;
    let bw_wmma = (total_bytes / (med_wmma * 1e-6)) / 1e9;

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<30} | {:<16} | {:<16} | {:<16} | {:<12}", "Metric", "FP16 GEMV (Base)", "FP16 MM-Tile", "Matrix Core WMMA", "Speedup WMMA");
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<30} | {:>13.2} us | {:>13.2} us | {:>13.2} us | {:>10.2}x",
             "Latency (Sync Median)", med_base, med_tile, med_wmma, med_base / med_wmma);
    println!("{:<30} | {:>13.2} GB/s| {:>13.2} GB/s| {:>13.2} GB/s| {:>10.2}x",
             "Effective Bandwidth", bw_base, bw_tile, bw_wmma, bw_wmma / bw_base);
    println!("-------------------------------------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_fp16_gemv_vs_mm_tile() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Shape 1: Qwen FFN Gate/Up Decode (M=1, N=17408, K=5120)
    run_f16_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Decode (M=1)",
        m: 1,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 2: Qwen FFN Down Decode (M=1, N=5120, K=17408)
    run_f16_bench(&BenchConfig {
        name: "Qwen FFN Down Decode (M=1)",
        m: 1,
        n: 5120,
        k: 17408,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 3: Batched Decode (M=4, N=17408, K=5120)
    run_f16_bench(&BenchConfig {
        name: "Qwen Batched Decode (M=4)",
        m: 4,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });
}
