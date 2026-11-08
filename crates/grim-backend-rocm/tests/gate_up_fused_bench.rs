//! Microbenchmark testing Fused Gate-Up + SwiGLU (Step 2) vs Unfused Gate + Up + SiLU:
//!   - Unfused: Gate GEMV + Up GEMV + Elementwise SiLU*Up
//!   - Fused: Single-launch `grim_fp32_gate_up_silu_mm_tile`
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test gate_up_fused_bench -- --nocapture

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

fn run_fused_gate_up_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;

    println!(
        "\n========================================================================================="
    );
    println!(
        "Unfused (Gate + Up + SiLU) vs Fused Gate-Up MM-Tile [{}]: M={}, N={}, K={}",
        cfg.name, m, n, k
    );
    println!(
        "========================================================================================="
    );

    let act: Vec<f32> = (0..(m * k)).map(|i| ((i % 50) as f32 + 1.0) * 0.02).collect();
    let gate_w: Vec<f32> = (0..(n * k)).map(|i| (((i % 17) as f32) - 8.0) * 0.005).collect();
    let up_w: Vec<f32> = (0..(n * k)).map(|i| (((i % 19) as f32) - 9.0) * 0.005).collect();

    let act_storage = upload_f32(&dev, &act, Shape::new(vec![m, k]));
    let gate_w_storage = upload_f32(&dev, &gate_w, Shape::new(vec![n, k]));
    let up_w_storage = upload_f32(&dev, &up_w, Shape::new(vec![n, k]));

    let gate_out_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let up_out_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let unfused_out_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));
    let fused_out_storage = upload_f32(&dev, &vec![0.0f32; m * n], Shape::new(vec![m, n]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let gate_w_s = gate_w_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let up_w_s = up_w_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    let gate_out_s = gate_out_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let up_out_s = up_out_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let _unfused_out_s = unfused_out_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let fused_out_s = fused_out_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Parity Check
    print!("[1/3] Parity check Unfused vs Fused Gate-Up MM-Tile... ");
    dev.launch_fp32_mm_tile_for_ab(act_s, gate_w_s, gate_out_s, m, n, k).unwrap();
    dev.launch_fp32_mm_tile_for_ab(act_s, up_w_s, up_out_s, m, n, k).unwrap();
    // Host compute unfused silu reference
    dev.synchronize();
    let g_res = gate_out_s.to_cpu_vec_f32().unwrap();
    let u_res = up_out_s.to_cpu_vec_f32().unwrap();
    let mut ref_out = vec![0.0f32; m * n];
    for i in 0..(m * n) {
        let g = g_res[i];
        let u = u_res[i];
        let silu_g = g / (1.0 + (-g).exp());
        ref_out[i] = silu_g * u;
    }

    dev.launch_fp32_gate_up_silu_mm_tile_for_ab(act_s, gate_w_s, up_w_s, fused_out_s, m, n, k).unwrap();
    dev.synchronize();
    let fused_res = fused_out_s.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..(m * n) {
        let d = (ref_out[i] - fused_res[i]).abs();
        if d > max_diff { max_diff = d; }
    }
    println!("OK (diff={:.2e})", max_diff);
    assert!(max_diff < 1e-4, "Parity check failed for Fused Gate-Up");

    // 2. Timing Unfused Path (Gate GEMV + Up GEMV)
    for _ in 0..cfg.warmup_iters {
        dev.launch_fp32_mm_tile_for_ab(act_s, gate_w_s, gate_out_s, m, n, k).unwrap();
        dev.launch_fp32_mm_tile_for_ab(act_s, up_w_s, up_out_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[2/3] Timing Unfused Gate GEMV + Up GEMV... ");
    let mut times_unfused: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_fp32_mm_tile_for_ab(act_s, gate_w_s, gate_out_s, m, n, k).unwrap();
        dev.launch_fp32_mm_tile_for_ab(act_s, up_w_s, up_out_s, m, n, k).unwrap();
        dev.synchronize();
        times_unfused.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Timing Fused Gate-Up MM-Tile
    for _ in 0..cfg.warmup_iters {
        dev.launch_fp32_gate_up_silu_mm_tile_for_ab(act_s, gate_w_s, up_w_s, fused_out_s, m, n, k).unwrap();
    }
    dev.synchronize();

    print!("[3/3] Timing Fused Gate-Up + SwiGLU MM-Tile... ");
    let mut times_fused: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.launch_fp32_gate_up_silu_mm_tile_for_ab(act_s, gate_w_s, up_w_s, fused_out_s, m, n, k).unwrap();
        dev.synchronize();
        times_fused.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_unfused.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_fused.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_unfused = times_unfused[cfg.bench_iters / 2];
    let med_fused = times_fused[cfg.bench_iters / 2];

    println!("\nComparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("---------------------------------------------------------------------------------------------------");
    println!("{:<32} | {:<20} | {:<20} | {:<12}", "Metric", "Unfused (Gate + Up)", "Fused Gate-Up SwiGLU", "Speedup");
    println!("---------------------------------------------------------------------------------------------------");
    println!("{:<32} | {:>17.2} us | {:>17.2} us | {:>10.2}x ({})",
             "Latency (Sync Median)", med_unfused, med_fused,
             med_unfused / med_fused,
             if med_fused < med_unfused { "Fused faster" } else { "Unfused faster" });
    println!("---------------------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_gate_up_fused() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Qwen FFN Gate/Up Decode (M=1, N=17408, K=5120)
    run_fused_gate_up_bench(&BenchConfig {
        name: "Qwen FFN Gate/Up Decode (M=1)",
        m: 1,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Qwen Batched Decode (M=4, N=17408, K=5120)
    run_fused_gate_up_bench(&BenchConfig {
        name: "Qwen Batched Decode (M=4)",
        m: 4,
        n: 17408,
        k: 5120,
        warmup_iters: 10,
        bench_iters: 40,
    });
}
