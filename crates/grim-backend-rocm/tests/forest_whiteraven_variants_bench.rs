//! Microbenchmark comparing baseline vs variants for:
//!   1. ForestRaven (INT8 / Q8_0):
//!      - Baseline: `grim_dot4_forest_gemv_v10`
//!      - Variant A: `grim_dot4_forest_gemv_v11_coop` (4-wave cooperative quantization + 128-bit vector loads)
//!      - Variant B: `grim_dot4_forest_gemv_rows_r2_v10` (2-row weight reuse for M=2)
//!   2. WhiteRaven (FP8 Blocked WMMA):
//!      - Baseline: `grim_wmma_gemm_fp8_e4m3_blocked_v2` (16x32 tile, 2x unroll)
//!      - Variant A: `grim_wmma_gemm_fp8_e4m3_blocked_v3_quad` (16x64 quad-tile output, 4x mma_sync reuse)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test forest_whiteraven_variants_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::{gpu_test_enabled, RocmDevice, RocmStorage, CoreTensorOps};
use grim_tensor::dtype::{ArithType, BlockDtype, DType, FloatPackScheme, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

fn upload_f32(dev: &RocmDevice, data: &[f32], shape: Shape) -> Box<dyn BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("f32 upload")
}

fn upload_forest_blob(dev: &RocmDevice, data: &[u8], n: usize, k: usize) -> Box<dyn BackendStorage> {
    dev.from_cpu_bytes(
        data,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Block(BlockDtype::Int8PerChannel),
        },
    )
    .expect("forest blob upload")
}

fn upload_fp8_blocked(dev: &RocmDevice, data: &[u8], shape: Shape) -> Box<dyn BackendStorage> {
    dev.from_cpu_bytes(
        data,
        &shape,
        DType {
            arith: ArithType::U8,
            storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8Blocked16),
        },
    )
    .expect("fp8 blocked upload")
}

struct BenchConfig {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

fn run_forest_variants_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");
    let (m, n, k) = (cfg.m, cfg.n, cfg.k);

    println!("\n=========================================================================================");
    println!("ForestRaven (INT8) Optimization Variants [{}]: M={}, N={}, K={}", cfg.name, m, n, k);
    println!("=========================================================================================");

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let weight_f32: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();

    // Pack ForestRaven framed blob: [8B codes_len][codes: i8][8B scales_len][scales: f32]
    let mut blob = Vec::with_capacity(16 + n * k + n * 4);
    let codes_len = (n * k) as u64;
    blob.extend_from_slice(&codes_len.to_le_bytes());
    let mut scales = Vec::with_capacity(n);
    for row in 0..n {
        let row_w = &weight_f32[row * k..(row + 1) * k];
        let mut amax = 0.0f32;
        for &w in row_w { if w.abs() > amax { amax = w.abs(); } }
        let scale = amax / 127.0;
        let inv_scale = if amax == 0.0 { 0.0 } else { 127.0 / amax };
        scales.push(scale);
        for &w in row_w {
            let q = (w * inv_scale).round().clamp(-127.0, 127.0) as i8;
            blob.push(q as u8);
        }
    }
    let scales_len = (n * 4) as u64;
    blob.extend_from_slice(&scales_len.to_le_bytes());
    for &s in &scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }

    let act_storage = upload_f32(&dev, &act, Shape::new(vec![m, k]));
    let blob_storage = upload_forest_blob(&dev, &blob, n, k);
    let out_base_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_var_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_var_ab_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let blob_s = blob_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_base_s = out_base_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_var_s = out_var_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_var_ab_s = out_var_ab_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // Parity check
    print!("[1/4] Parity check Baseline vs Variants... ");
    dev.launch_dot4_forest_gemv_v10_for_ab(act_s, blob_s, out_base_s, m, n, k).unwrap();
    if m == 1 {
        dev.launch_dot4_forest_gemv_v11_coop_for_ab(act_s, blob_s, out_var_s, m, n, k).unwrap();
    } else if m <= 2 {
        dev.launch_dot4_forest_gemv_rows_r2_v10_for_ab(act_s, blob_s, out_var_s, m, n, k).unwrap();
        dev.launch_dot4_forest_gemv_rows_r2_v11_coop_for_ab(act_s, blob_s, out_var_ab_s, m, n, k).unwrap();
    }
    dev.synchronize();

    let base_res = out_base_s.to_cpu_vec_f32().unwrap();
    let var_res = out_var_s.to_cpu_vec_f32().unwrap();
    let mut max_diff = 0.0f32;
    for i in 0..base_res.len() {
        let diff = (base_res[i] - var_res[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    if m <= 2 && m > 1 {
        let var_ab_res = out_var_ab_s.to_cpu_vec_f32().unwrap();
        for i in 0..base_res.len() {
            let diff = (base_res[i] - var_ab_res[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
    }
    println!("OK (diff={:.2e})", max_diff);

    // Timing Baseline
    for _ in 0..cfg.warmup_iters {
        dev.launch_dot4_forest_gemv_v10_for_ab(act_s, blob_s, out_base_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/4] Timing ForestRaven Baseline (v10)... ");
    let mut times_base = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_dot4_forest_gemv_v10_for_ab(act_s, blob_s, out_base_s, m, n, k).unwrap();
        dev.synchronize();
        times_base.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // Timing Variant
    let var_name = if m == 1 { "Variant A (v11_coop + uint4)" } else { "Variant B (rows_r2_v10)" };
    for _ in 0..cfg.warmup_iters {
        if m == 1 {
            dev.launch_dot4_forest_gemv_v11_coop_for_ab(act_s, blob_s, out_var_s, m, n, k).unwrap();
        } else {
            dev.launch_dot4_forest_gemv_rows_r2_v10_for_ab(act_s, blob_s, out_var_s, m, n, k).unwrap();
        }
    }
    dev.synchronize();

    print!("[3/4] Timing ForestRaven {}... ", var_name);
    let mut times_var = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        if m == 1 {
            dev.launch_dot4_forest_gemv_v11_coop_for_ab(act_s, blob_s, out_var_s, m, n, k).unwrap();
        } else {
            dev.launch_dot4_forest_gemv_rows_r2_v10_for_ab(act_s, blob_s, out_var_s, m, n, k).unwrap();
        }
        dev.synchronize();
        times_var.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // Timing Variant A+B (for M=2)
    let mut times_var_ab = Vec::with_capacity(cfg.bench_iters);
    if m <= 2 && m > 1 {
        for _ in 0..cfg.warmup_iters {
            dev.launch_dot4_forest_gemv_rows_r2_v11_coop_for_ab(act_s, blob_s, out_var_ab_s, m, n, k).unwrap();
        }
        dev.synchronize();

        print!("[4/4] Timing ForestRaven Variant A+B (rows_r2 + uint4 + coop)... ");
        for _ in 0..cfg.bench_iters {
            let t0 = Instant::now();
            dev.launch_dot4_forest_gemv_rows_r2_v11_coop_for_ab(act_s, blob_s, out_var_ab_s, m, n, k).unwrap();
            dev.synchronize();
            times_var_ab.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        println!("DONE");
    }

    times_base.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_var.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_var_ab.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_base = times_base[cfg.bench_iters / 2];
    let med_var = times_var[cfg.bench_iters / 2];
    let total_bytes = (n * k * 1 + m * k * 4 + m * n * 4) as f64;

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<36} | {:<16} | {:<16} | {:<12}", "Kernel", "Latency (Sync Med)", "Effective Bandwidth", "Per-Token");
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<36} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us", "ForestRaven Baseline (v10)", med_base, (total_bytes / (med_base * 1e-6)) / 1e9, med_base / (m as f64));
    println!("{:<36} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us", var_name, med_var, (total_bytes / (med_var * 1e-6)) / 1e9, med_var / (m as f64));
    if m <= 2 && m > 1 {
        let med_ab = times_var_ab[cfg.bench_iters / 2];
        println!("{:<36} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us", "Variant A+B (Combined)", med_ab, (total_bytes / (med_ab * 1e-6)) / 1e9, med_ab / (m as f64));
        println!("-------------------------------------------------------------------------------------------------------------------");
        println!("Speedup Variant A+B vs Baseline: {:.2}x\n", med_base / med_ab);
    } else {
        println!("-------------------------------------------------------------------------------------------------------------------");
        println!("Speedup vs Baseline: {:.2}x\n", med_base / med_var);
    }
}

fn run_whiteraven_variants_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");
    let (m, n, k) = (cfg.m, cfg.n, cfg.k);

    println!("\n=========================================================================================");
    println!("WhiteRaven (FP8 WMMA) Optimization Variants [{}]: M={}, N={}, K={}", cfg.name, m, n, k);
    println!("=========================================================================================");

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let weight_f32: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();
    let weight_fp8: Vec<u8> = weight_f32.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked_b = grim_quant::block_fp8_16x16(&weight_fp8, n, k).expect("block_fp8_16x16");

    let act_storage = upload_f32(&dev, &act, Shape::new(vec![m, k]));
    let weight_storage = upload_fp8_blocked(&dev, &blocked_b, Shape::new(vec![n, k]));
    let out_v2_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_v3_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let weight_s = weight_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v2_s = out_v2_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_v3_s = out_v3_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    let a_rows = m.div_ceil(16) * 16;
    let scratch = dev.white_raven_act_scratch(a_rows * k).unwrap();
    dev.launch_quant_fp8_pad16(act_s, &scratch, m, k).unwrap();

    // Parity check
    print!("[1/3] Parity check WhiteRaven v2 (Baseline) vs v3_quad (Variant A)... ");
    dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(scratch.as_ref(), weight_s, out_v2_s, m, n, k).unwrap();
    dev.launch_wmma_gemm_fp8_e4m3_blocked_v3_quad_for_ab(scratch.as_ref(), weight_s, out_v3_s, m, n, k).unwrap();
    dev.synchronize();

    let v2_res = out_v2_s.to_cpu_vec_f32().unwrap();
    let v3_res = out_v3_s.to_cpu_vec_f32().unwrap();
    let mut max_diff = 0.0f32;
    for i in 0..v2_res.len() {
        let diff = (v2_res[i] - v3_res[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("OK (diff={:.2e})", max_diff);

    // Timing v2
    for _ in 0..cfg.warmup_iters {
        dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(scratch.as_ref(), weight_s, out_v2_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/3] Timing WhiteRaven v2 (Baseline)... ");
    let mut times_v2 = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(scratch.as_ref(), weight_s, out_v2_s, m, n, k).unwrap();
        dev.synchronize();
        times_v2.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // Timing v3_quad
    for _ in 0..cfg.warmup_iters {
        dev.launch_wmma_gemm_fp8_e4m3_blocked_v3_quad_for_ab(scratch.as_ref(), weight_s, out_v3_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[3/3] Timing WhiteRaven v3_quad (16x64 quad-tile)... ");
    let mut times_v3 = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_wmma_gemm_fp8_e4m3_blocked_v3_quad_for_ab(scratch.as_ref(), weight_s, out_v3_s, m, n, k).unwrap();
        dev.synchronize();
        times_v3.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_v2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_v3.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_v2 = times_v2[cfg.bench_iters / 2];
    let med_v3 = times_v3[cfg.bench_iters / 2];
    let total_bytes = (n * k * 1 + m * k * 1 + m * n * 4) as f64;

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<32} | {:<16} | {:<16} | {:<12}", "Kernel", "Latency (Sync Med)", "Effective Bandwidth", "Per-Token");
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<32} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us", "WhiteRaven v2 (Baseline)", med_v2, (total_bytes / (med_v2 * 1e-6)) / 1e9, med_v2 / (m as f64));
    println!("{:<32} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us", "WhiteRaven v3_quad (Variant A)", med_v3, (total_bytes / (med_v3 * 1e-6)) / 1e9, med_v3 / (m as f64));
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("Speedup: {:.2}x\n", med_v2 / med_v3);
}

#[test]
fn bench_forest_and_whiteraven_variants() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // ForestRaven Shape 1: Decode M=1
    run_forest_variants_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Decode (M=1)",
        m: 1,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // ForestRaven Shape 2: Dual Decode M=2
    run_forest_variants_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Dual Decode (M=2)",
        m: 2,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // WhiteRaven Shape 1: Prefill M=16
    run_whiteraven_variants_bench(&BenchConfig {
        name: "Qwen Prefill Tile (M=16)",
        m: 16,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // WhiteRaven Shape 2: Large Prefill M=64
    run_whiteraven_variants_bench(&BenchConfig {
        name: "Qwen Batched Prefill (M=64)",
        m: 64,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });
}
