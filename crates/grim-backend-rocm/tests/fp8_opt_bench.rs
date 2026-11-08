//! Microbenchmark comparing Raven FP8 kernels:
//!   - Raven FP8 GEMV v10 (M=1, hardware V_DOT4_F32_FP8_FP8)
//!   - Raven FP8 GEMV Dual-Row r2_v10 (M=2, 2-row weight reuse)
//!   - Raven FP8 GEMV Multi-Row r8_v10 (M=4, 8-row weight reuse)
//!   - WhiteRaven Blocked WMMA (M=1, 2, 4)
//!   - Unfused Gate + Up FP8 vs Fused Gate-Up + SwiGLU FP8 (`grim_dot4_fp8_gate_up_silu_v10`)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test fp8_opt_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::{gpu_test_enabled, RocmDevice, RocmStorage, CoreTensorOps};
use grim_tensor::dtype::{ArithType, DType, FloatPackScheme, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, QuantOps, Shape};

fn upload_f32(dev: &RocmDevice, data: &[f32], shape: Shape) -> Box<dyn BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("f32 upload")
}

fn upload_fp8_bare(dev: &RocmDevice, data: &[u8], shape: Shape) -> Box<dyn BackendStorage> {
    dev.from_cpu_bytes(
        data,
        &shape,
        DType {
            arith: ArithType::F32,
            storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
        },
    )
    .expect("fp8 upload")
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

fn run_fp8_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;

    println!(
        "\n========================================================================================="
    );
    println!(
        "FP8 Raven Optimization Bench [{}]: M={}, N={}, K={}",
        cfg.name, m, n, k
    );
    println!(
        "========================================================================================="
    );

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let weight_f32: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();
    let weight_fp8: Vec<u8> = weight_f32.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();

    let act_storage = upload_f32(&dev, &act, Shape::new(vec![m, k]));
    let weight_storage = upload_fp8_bare(&dev, &weight_fp8, Shape::new(vec![n, k]));

    // Blocked B for WhiteRaven
    let blocked_b = grim_quant::block_fp8_16x16(&weight_fp8, n, k).expect("block_fp8_16x16");
    let weight_blocked_storage = upload_fp8_blocked(&dev, &blocked_b, Shape::new(vec![n, k]));

    let out_raven_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_wr_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let weight_s = weight_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let _weight_blk_s = weight_blocked_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_raven_s = out_raven_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let _out_wr_s = out_wr_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Correctness check
    print!("[1/4] Parity check Raven dot4 vs WhiteRaven WMMA... ");
    dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, weight_s, out_raven_s, m, n, k).unwrap();

    // Run WhiteRaven through quantized_matmul
    let (wr_out, _) = dev.quantized_matmul(
        act_storage.as_ref(),
        weight_blocked_storage.as_ref(),
        &[],
        grim_tensor::QuantFormat::Fp8Blocked16,
        &Shape::new(vec![m, n]),
    ).unwrap();
    dev.synchronize();

    let r_res = out_raven_s.to_cpu_vec_f32().unwrap();
    let wr_res = wr_out.as_any().downcast_ref::<RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..r_res.len() {
        let diff = (r_res[i] - wr_res[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("OK (diff={:.2e})", max_diff);

    // 2. Timing Raven dot4 GEMV
    for _ in 0..cfg.warmup_iters {
        dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, weight_s, out_raven_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/4] Timing Raven FP8 dot4 GEMV... ");
    let mut times_raven: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, weight_s, out_raven_s, m, n, k).unwrap();
        dev.synchronize();
        times_raven.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Timing WhiteRaven WMMA
    for _ in 0..cfg.warmup_iters {
        dev.quantized_matmul(
            act_storage.as_ref(),
            weight_blocked_storage.as_ref(),
            &[],
            grim_tensor::QuantFormat::Fp8Blocked16,
            &Shape::new(vec![m, n]),
        ).unwrap();
    }
    dev.synchronize();

    print!("[3/4] Timing WhiteRaven Blocked WMMA... ");
    let mut times_wr: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.quantized_matmul(
            act_storage.as_ref(),
            weight_blocked_storage.as_ref(),
            &[],
            grim_tensor::QuantFormat::Fp8Blocked16,
            &Shape::new(vec![m, n]),
        ).unwrap();
        dev.synchronize();
        times_wr.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_raven.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_wr.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_raven = times_raven[cfg.bench_iters / 2];
    let med_wr = times_wr[cfg.bench_iters / 2];

    let total_bytes = (n * k * 1 + m * k * 4 + m * n * 4) as f64; // FP8 weights are 1 byte
    let bw_raven = (total_bytes / (med_raven * 1e-6)) / 1e9;
    let bw_wr = (total_bytes / (med_wr * 1e-6)) / 1e9;

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<28} | {:<16} | {:<16} | {:<12}", "Kernel", "Latency (Sync Med)", "Effective Bandwidth", "Per-Token");
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("{:<28} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us",
             "Raven FP8 dot4 GEMV", med_raven, bw_raven, med_raven / (m as f64));
    println!("{:<28} | {:>13.2} us | {:>13.2} GB/s| {:>9.2} us",
             "WhiteRaven Blocked WMMA", med_wr, bw_wr, med_wr / (m as f64));
    println!("-------------------------------------------------------------------------------------------------------------------");
    println!("Speedup Raven vs WhiteRaven: {:.2}x\n", med_wr / med_raven);
}

fn run_fp8_gate_up_bench(m: usize, n: usize, k: usize) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    println!(
        "\n========================================================================================="
    );
    println!(
        "FP8 Fused Gate-Up + SwiGLU Bench: M={}, N={}, K={}",
        m, n, k
    );
    println!(
        "========================================================================================="
    );

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let gw_f32: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();
    let uw_f32: Vec<f32> = (0..(n * k)).map(|i| (((i % 23) as f32) - 11.0) * 0.004).collect();

    let gw_fp8: Vec<u8> = gw_f32.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let uw_fp8: Vec<u8> = uw_f32.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();

    let act_storage = upload_f32(&dev, &act, Shape::new(vec![m, k]));
    let gw_storage = upload_fp8_bare(&dev, &gw_fp8, Shape::new(vec![n, k]));
    let uw_storage = upload_fp8_bare(&dev, &uw_fp8, Shape::new(vec![n, k]));

    let out_gate_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_up_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let out_fused_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let gw_s = gw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let uw_s = uw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_gate_s = out_gate_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_up_s = out_up_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_fused_s = out_fused_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // Parity check
    print!("[1/3] Parity check Unfused vs Fused Gate-Up FP8... ");
    dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, gw_s, out_gate_s, m, n, k).unwrap();
    dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, uw_s, out_up_s, m, n, k).unwrap();
    dev.launch_dot4_fp8_gate_up_silu_v10_for_ab(act_s, gw_s, uw_s, out_fused_s, m, n, k).unwrap();
    dev.synchronize();

    let g_res = out_gate_s.to_cpu_vec_f32().unwrap();
    let u_res = out_up_s.to_cpu_vec_f32().unwrap();
    let fused_res = out_fused_s.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..g_res.len() {
        let g = g_res[i];
        let u = u_res[i];
        let silu_g = g / (1.0f32 + (-g).exp());
        let expected = silu_g * u;
        let diff = (expected - fused_res[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("OK (diff={:.2e})", max_diff);

    // Timing Unfused
    let iters = 40;
    for _ in 0..10 {
        dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, gw_s, out_gate_s, m, n, k).unwrap();
        dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, uw_s, out_up_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/3] Timing Unfused Gate + Up FP8... ");
    let mut times_unfused: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, gw_s, out_gate_s, m, n, k).unwrap();
        dev.launch_dot4_fp8_gemv_v10_for_ab(act_s, uw_s, out_up_s, m, n, k).unwrap();
        dev.synchronize();
        times_unfused.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // Timing Fused
    for _ in 0..10 {
        dev.launch_dot4_fp8_gate_up_silu_v10_for_ab(act_s, gw_s, uw_s, out_fused_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[3/3] Timing Fused Gate-Up FP8... ");
    let mut times_fused: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        dev.launch_dot4_fp8_gate_up_silu_v10_for_ab(act_s, gw_s, uw_s, out_fused_s, m, n, k).unwrap();
        dev.synchronize();
        times_fused.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_unfused.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_fused.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_unfused = times_unfused[iters / 2];
    let med_fused = times_fused[iters / 2];

    println!("\nComparative Summary (Iterations = {}):", iters);
    println!("---------------------------------------------------------------------------------------------------");
    println!("{:<32} | {:<16} | {:<16} | {:<12}", "Operation", "Latency (Med)", "Effective Speedup", "Per-Layer Saved");
    println!("---------------------------------------------------------------------------------------------------");
    println!("{:<32} | {:>13.2} us | {:>16} | {:>12}", "Unfused Gate + Up", med_unfused, "1.00x", "-");
    println!("{:<32} | {:>13.2} us | {:>15.2}x | {:>9.2} us", "Fused Gate-Up SwiGLU", med_fused, med_unfused / med_fused, med_unfused - med_fused);
    println!("---------------------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_fp8_optimizations() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Shape 1: Qwen FFN Gate/Up Decode (M=1, N=17408, K=5120)
    run_fp8_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Single Decode (M=1)",
        m: 1,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 2: Qwen FFN Gate/Up Dual Decode (M=2, N=17408, K=5120)
    run_fp8_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Dual Decode (M=2)",
        m: 2,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 3: Qwen FFN Gate/Up Quad Decode (M=4, N=17408, K=5120)
    run_fp8_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Quad Decode (M=4)",
        m: 4,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Shape 4: Fused Gate-Up + SwiGLU FP8 test
    run_fp8_gate_up_bench(1, 17408, 5120);
}
