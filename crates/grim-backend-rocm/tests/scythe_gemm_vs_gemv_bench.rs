//! Microbenchmark comparing Scythe Ring GEMV (scalar FMA) vs Scythe Ring WMMA (tensor-core GEMM):
//!   - Mode A (GEMV): `grim_moe_fused_grouped` (scalar dot-products)
//!   - Mode B (GEMM): `grim_moe_fused_grouped_wmma` (rocWMMA 16x16 tensor-core tiles)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test scythe_gemm_vs_gemv_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::kernels::charon::{moe_align_block_size, RoutingAssignment};
use grim_backend_rocm::{as_rocm, dev_ptr, gpu_test_enabled, RocmDevice};
use grim_tensor::dtype::DType;
use grim_tensor::{BackendStorage, CoreTensorOps, MemoryOps, Shape};

struct BenchConfig {
    name: &'static str,
    batch: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    block_size: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

fn run_gemv_vs_gemm_bench(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let batch = cfg.batch;
    let hidden = cfg.hidden;
    let inter = cfg.inter;
    let num_experts = cfg.num_experts;
    let top_k = cfg.top_k;
    let num_pairs = batch * top_k;
    let block_size = cfg.block_size;

    println!(
        "\n========================================================================================="
    );
    println!(
        "Scythe MoE GEMV vs GEMM Benchmark [{}]: batch={}, hidden={}, inter={}, experts={}, top_k={}",
        cfg.name, batch, hidden, inter, num_experts, top_k
    );
    println!(
        "========================================================================================="
    );

    let activations: Vec<f32> = (0..(batch * hidden))
        .map(|i| ((i % 100) as f32 + 1.0) * 0.01)
        .collect();
    let expert_gate_w: Vec<f32> = (0..(num_experts * inter * hidden))
        .map(|i| (((i % 13) as f32) - 6.0) * 0.005)
        .collect();
    let expert_up_w: Vec<f32> = (0..(num_experts * inter * hidden))
        .map(|i| (((i % 11) as f32) - 5.0) * 0.005)
        .collect();
    let expert_down_w: Vec<f32> = (0..(num_experts * hidden * inter))
        .map(|i| (((i % 7) as f32) - 3.0) * 0.005)
        .collect();

    let mut token_ids: Vec<u32> = Vec::with_capacity(num_pairs);
    let mut expert_ids: Vec<u32> = Vec::with_capacity(num_pairs);
    let mut weights: Vec<f32> = Vec::with_capacity(num_pairs);

    for t in 0..batch {
        for k in 0..top_k {
            token_ids.push(t as u32);
            expert_ids.push(((t * top_k + k) % num_experts) as u32);
            weights.push(1.0 / (top_k as f32));
        }
    }

    let assignment = RoutingAssignment {
        tokens: token_ids,
        experts: expert_ids,
        weights,
    };

    let sorted = moe_align_block_size(&assignment, block_size, num_experts);

    let act_shape = Shape::new(vec![batch, hidden]);
    let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
    let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
    let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
    let out_shape = Shape::new(vec![batch, hidden]);

    let act_storage: Box<dyn BackendStorage> =
        CoreTensorOps::from_cpu(&dev, &activations, &act_shape, DType::F32).unwrap();
    let gw_storage: Box<dyn BackendStorage> =
        CoreTensorOps::from_cpu(&dev, &expert_gate_w, &exp_gate_shape, DType::F32).unwrap();
    let uw_storage: Box<dyn BackendStorage> =
        CoreTensorOps::from_cpu(&dev, &expert_up_w, &exp_up_shape, DType::F32).unwrap();
    let dw_storage: Box<dyn BackendStorage> =
        CoreTensorOps::from_cpu(&dev, &expert_down_w, &exp_down_shape, DType::F32).unwrap();

    let out_gemv_storage: Box<dyn BackendStorage> =
        MemoryOps::alloc_storage(&dev, &out_shape, DType::F32).unwrap();
    let out_gemm_storage: Box<dyn BackendStorage> =
        MemoryOps::alloc_storage(&dev, &out_shape, DType::F32).unwrap();

    let act_s = as_rocm(act_storage.as_ref()).unwrap();
    let gw_s = as_rocm(gw_storage.as_ref()).unwrap();
    let uw_s = as_rocm(uw_storage.as_ref()).unwrap();
    let dw_s = as_rocm(dw_storage.as_ref()).unwrap();
    let out_gemv_s = as_rocm(out_gemv_storage.as_ref()).unwrap();
    let out_gemm_s = as_rocm(out_gemm_storage.as_ref()).unwrap();

    let gw_ptr = dev_ptr(gw_s).unwrap();
    let uw_ptr = dev_ptr(uw_s).unwrap();
    let dw_ptr = dev_ptr(dw_s).unwrap();

    // Verification run
    print!("[1/4] Running numerical parity check (GEMV vs WMMA GEMM)... ");
    let _ = dev
        .launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_gemv_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped",
            None,
            None,
        )
        .expect("GEMV launch");

    let _ = dev
        .launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_gemm_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        )
        .expect("WMMA GEMM launch");

    dev.synchronize();

    let res_gemv = out_gemv_storage.to_cpu_vec_f32().unwrap();
    let res_gemm = out_gemm_storage.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for (v_v, v_m) in res_gemv.iter().zip(res_gemm.iter()) {
        let diff = (v_v - v_m).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("OK (max diff: {:.6e})", max_diff);
    assert!(
        max_diff < 1e-4,
        "Parity check failed: max diff {:.6e} >= 1e-4",
        max_diff
    );

    // Warmup GEMV
    for _ in 0..cfg.warmup_iters {
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_gemv_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped",
            None,
            None,
        );
    }
    dev.synchronize();

    // Benchmark Mode A (GEMV)
    print!("[2/4] Benchmarking Scythe GEMV mode (scalar FMA)... ");
    let mut gemv_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_gemv_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped",
            None,
            None,
        );
        dev.synchronize();
        gemv_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // Warmup WMMA GEMM
    for _ in 0..cfg.warmup_iters {
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_gemm_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        );
    }
    dev.synchronize();

    // Benchmark Mode B (WMMA GEMM)
    print!("[3/4] Benchmarking Scythe WMMA GEMM mode (tensor-core tiles)... ");
    let mut gemm_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_gemm_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        );
        dev.synchronize();
        gemm_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    gemv_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gemm_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let gemv_median_us = gemv_times_us[cfg.bench_iters / 2];
    let gemm_median_us = gemm_times_us[cfg.bench_iters / 2];

    let total_flops = (num_pairs as f64) * (6.0 * hidden as f64 * inter as f64);
    let gemv_tflops = (total_flops / (gemv_median_us * 1e-6)) / 1e12;
    let gemm_tflops = (total_flops / (gemm_median_us * 1e-6)) / 1e12;

    let gemv_tokens_per_sec = (batch as f64) / (gemv_median_us * 1e-6);
    let gemm_tokens_per_sec = (batch as f64) / (gemm_median_us * 1e-6);

    println!("[4/4] Comparative Performance Summary:");
    println!(
        "\nResults summary ({} iters, per dispatch):",
        cfg.bench_iters
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:<16} | {:<16} | {:<14}",
        "Metric", "Scythe GEMV", "Scythe WMMA GEMM", "Speedup"
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:>13.2} us | {:>13.2} us | {:>12.2}x ({})",
        "Kernel Latency (Sync Median)",
        gemv_median_us,
        gemm_median_us,
        gemv_median_us / gemm_median_us,
        if gemm_median_us < gemv_median_us {
            "WMMA GEMM faster"
        } else {
            "GEMV faster"
        }
    );
    println!(
        "{:<36} | {:>13.4} T | {:>13.4} T | {:>12.2}x",
        "Throughput (Compute TFLOPS)",
        gemv_tflops,
        gemm_tflops,
        gemm_tflops / gemv_tflops
    );
    println!(
        "{:<36} | {:>13.0}   | {:>13.0}   | {:>12.2}x",
        "Throughput (Tokens / sec)",
        gemv_tokens_per_sec,
        gemm_tokens_per_sec,
        gemm_tokens_per_sec / gemv_tokens_per_sec
    );
    println!("-----------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_scythe_gemm_vs_gemv() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Config 1: Production Decode Shape (batch=1, hidden=2048, inter=512, experts=8, top_k=2)
    run_gemv_vs_gemm_bench(&BenchConfig {
        name: "Single Decode (M=1)",
        batch: 1,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 2,
        warmup_iters: 5,
        bench_iters: 20,
    });

    // Config 2: Batched Decode Shape (batch=4, hidden=2048, inter=512, experts=8, top_k=2)
    run_gemv_vs_gemm_bench(&BenchConfig {
        name: "Batched Decode (M=4)",
        batch: 4,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 8,
        warmup_iters: 5,
        bench_iters: 20,
    });

    // Config 3: Prefill Tile Shape (batch=16, hidden=2048, inter=512, experts=8, top_k=2)
    run_gemv_vs_gemm_bench(&BenchConfig {
        name: "Prefill Tile (M=16)",
        batch: 16,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 16,
        warmup_iters: 5,
        bench_iters: 20,
    });
}
