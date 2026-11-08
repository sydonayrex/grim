//! Standalone microbenchmark comparing throughput, kernel busy time, and wall latency:
//!   - Path A: Charon resident MoE (`moe_fused_dispatch_resident_routing_into`)
//!   - Path B: Scythe dynamic MoE (`enqueue_dynamic_moe`)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test moe_dispatch_scythe_ab_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::kernels::task_ring::TaskRingContext;
use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
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

fn upload_u32(dev: &RocmDevice, data: &[u32], shape: Shape) -> Box<dyn BackendStorage> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    dev.from_cpu_bytes(
        &bytes,
        &shape,
        DType {
            arith: ArithType::U32,
            storage: DTypeStorage::Native,
        },
    )
    .expect("u32 upload")
}

struct BenchConfig {
    name: &'static str,
    num_tokens: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    block_size: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

fn run_ab_benchmark_config(cfg: &BenchConfig) {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let num_tokens = cfg.num_tokens;
    let hidden = cfg.hidden;
    let inter = cfg.inter;
    let num_experts = cfg.num_experts;
    let top_k = cfg.top_k;
    let num_pairs = num_tokens * top_k;
    let block_size = cfg.block_size;

    println!(
        "\n================================================================================"
    );
    println!(
        "MoE A/B Microbenchmark [{}]: tokens={}, hidden={}, inter={}, experts={}, top_k={}",
        cfg.name, num_tokens, hidden, inter, num_experts, top_k
    );
    println!(
        "================================================================================"
    );

    // Activations: [num_tokens, hidden]
    let x_data: Vec<f32> = (0..(num_tokens * hidden))
        .map(|i| ((i % 100) as f32 + 1.0) * 0.01)
        .collect();
    let gate_w_data: Vec<f32> = (0..(num_experts * inter * hidden))
        .map(|i| (((i % 13) as f32) - 6.0) * 0.005)
        .collect();
    let up_w_data: Vec<f32> = (0..(num_experts * inter * hidden))
        .map(|i| (((i % 11) as f32) - 5.0) * 0.005)
        .collect();
    let down_w_data: Vec<f32> = (0..(num_experts * hidden * inter))
        .map(|i| (((i % 7) as f32) - 3.0) * 0.005)
        .collect();

    let mut token_ids: Vec<u32> = Vec::with_capacity(num_pairs);
    let mut expert_ids: Vec<u32> = Vec::with_capacity(num_pairs);
    let mut weights: Vec<f32> = Vec::with_capacity(num_pairs);

    for t in 0..num_tokens {
        for k in 0..top_k {
            token_ids.push(t as u32);
            expert_ids.push(((t * top_k + k) % num_experts) as u32);
            weights.push(1.0 / (top_k as f32));
        }
    }

    let x_storage = upload_f32(&dev, &x_data, Shape::new(vec![num_tokens, hidden]));
    let gw_storage = upload_f32(
        &dev,
        &gate_w_data,
        Shape::new(vec![num_experts, inter, hidden]),
    );
    let uw_storage = upload_f32(&dev, &up_w_data, Shape::new(vec![num_experts, inter, hidden]));
    let dw_storage = upload_f32(
        &dev,
        &down_w_data,
        Shape::new(vec![num_experts, hidden, inter]),
    );

    let ti_storage = upload_u32(&dev, &token_ids, Shape::new(vec![num_pairs]));
    let ei_storage = upload_u32(&dev, &expert_ids, Shape::new(vec![num_pairs]));
    let w_storage = upload_f32(&dev, &weights, Shape::new(vec![num_pairs]));

    let out_a_storage = upload_f32(
        &dev,
        &vec![0.0f32; num_tokens * hidden],
        Shape::new(vec![num_tokens, hidden]),
    );
    let out_b_storage = upload_f32(
        &dev,
        &vec![0.0f32; num_tokens * hidden],
        Shape::new(vec![num_tokens, hidden]),
    );

    let x_rocm = x_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ti_rocm = ti_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ei_rocm = ei_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let w_rocm = w_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_a_rocm = out_a_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_b_rocm = out_b_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    let x_ptr = x_rocm.device_ptr_u64().unwrap();
    let gw_ptr = gw_storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap()
        .device_ptr_u64()
        .unwrap();
    let uw_ptr = uw_storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap()
        .device_ptr_u64()
        .unwrap();
    let dw_ptr = dw_storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap()
        .device_ptr_u64()
        .unwrap();
    let ti_ptr = ti_rocm.device_ptr_u64().unwrap();
    let ei_ptr = ei_rocm.device_ptr_u64().unwrap();
    let w_ptr = w_rocm.device_ptr_u64().unwrap();
    let out_b_ptr = out_b_rocm.device_ptr_u64().unwrap();

    let mut graph_b = TaskGraphBuilder::new();
    let h_in = graph_b.push_arg(x_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_IN);
    let h_out = graph_b.push_arg(out_b_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_OUT);
    let h_gw = graph_b.push_arg(gw_ptr, (gate_w_data.len() * 4) as u64, arg_role::WEIGHT);
    let h_uw = graph_b.push_arg(uw_ptr, (up_w_data.len() * 4) as u64, arg_role::WEIGHT);
    let h_dw = graph_b.push_arg(dw_ptr, (down_w_data.len() * 4) as u64, arg_role::WEIGHT);
    let h_ti = graph_b.push_arg(ti_ptr, (num_pairs * 4) as u64, arg_role::SCRATCH);
    let h_ei = graph_b.push_arg(ei_ptr, (num_pairs * 4) as u64, arg_role::SCRATCH);
    let h_w = graph_b.push_arg(w_ptr, (num_pairs * 4) as u64, arg_role::SCRATCH);

    let params = [
        0u32, // quant_mode = 0 (FP32)
        hidden as u32,
        inter as u32,
        num_tokens as u32,
        block_size as u32,
        num_experts as u32,
        top_k as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    graph_b.push(
        op::MOE_DISPATCH,
        params,
        &[h_in, h_out, h_gw, h_uw, h_dw, h_ti, h_ei, h_w],
        &[],
        0,
        0,
    );
    let graph = graph_b.finish();
    let ring_ctx = TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context creation")
        .expect("ring binds moes");
    let moe_binding = &ring_ctx.plan().moes()[0];

    // Verification run
    print!("[1/4] Running parity verification run... ");
    dev.moe_fused_dispatch_resident_routing_into(
        x_rocm,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_rocm,
        ei_rocm,
        w_rocm,
        num_pairs,
        out_a_rocm,
        hidden,
        inter,
        1.0,
    )
    .expect("Path A verification run");

    ring_ctx
        .enqueue_dynamic_moe(&dev, moe_binding)
        .expect("Path B verification run");

    dev.synchronize();

    let res_a = out_a_rocm.to_cpu_vec_f32().unwrap();
    let res_b = out_b_rocm.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for (va, vb) in res_a.iter().zip(res_b.iter()) {
        let diff = (va - vb).abs();
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

    // Warmup Path A
    for _ in 0..cfg.warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into(
            x_rocm,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_rocm,
            ei_rocm,
            w_rocm,
            num_pairs,
            out_a_rocm,
            hidden,
            inter,
            1.0,
        )
        .unwrap();
    }
    dev.synchronize();

    // Benchmark Path A: Synchronous single iteration kernel busy time
    print!("[2/4] Benchmarking Path A (Charon resident)... ");
    let mut path_a_sync_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.moe_fused_dispatch_resident_routing_into(
            x_rocm,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_rocm,
            ei_rocm,
            w_rocm,
            num_pairs,
            out_a_rocm,
            hidden,
            inter,
            1.0,
        )
        .unwrap();
        dev.synchronize();
        path_a_sync_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    // Benchmark Path A: Host dispatch wall latency
    let t_async_a = Instant::now();
    for _ in 0..cfg.bench_iters {
        dev.moe_fused_dispatch_resident_routing_into(
            x_rocm,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_rocm,
            ei_rocm,
            w_rocm,
            num_pairs,
            out_a_rocm,
            hidden,
            inter,
            1.0,
        )
        .unwrap();
    }
    let path_a_launch_wall_us = (t_async_a.elapsed().as_nanos() as f64 / 1000.0) / (cfg.bench_iters as f64);
    dev.synchronize();
    let path_a_total_wall_us = (t_async_a.elapsed().as_nanos() as f64 / 1000.0) / (cfg.bench_iters as f64);
    println!("DONE");

    // Warmup Path B
    for _ in 0..cfg.warmup_iters {
        ring_ctx.enqueue_dynamic_moe(&dev, moe_binding).unwrap();
    }
    dev.synchronize();

    // Benchmark Path B: Synchronous single iteration kernel busy time
    print!("[3/4] Benchmarking Path B (Scythe dynamic ring)... ");
    let mut path_b_sync_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        ring_ctx.enqueue_dynamic_moe(&dev, moe_binding).unwrap();
        dev.synchronize();
        path_b_sync_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    // Benchmark Path B: Host dispatch wall latency
    let t_async_b = Instant::now();
    for _ in 0..cfg.bench_iters {
        ring_ctx.enqueue_dynamic_moe(&dev, moe_binding).unwrap();
    }
    let path_b_launch_wall_us = (t_async_b.elapsed().as_nanos() as f64 / 1000.0) / (cfg.bench_iters as f64);
    dev.synchronize();
    let path_b_total_wall_us = (t_async_b.elapsed().as_nanos() as f64 / 1000.0) / (cfg.bench_iters as f64);
    println!("DONE");

    path_a_sync_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    path_b_sync_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let path_a_median_busy_us = path_a_sync_times_us[cfg.bench_iters / 2];
    let path_b_median_busy_us = path_b_sync_times_us[cfg.bench_iters / 2];

    let total_flops = (num_pairs as f64) * (6.0 * hidden as f64 * inter as f64);
    let path_a_tflops = (total_flops / (path_a_median_busy_us * 1e-6)) / 1e12;
    let path_b_tflops = (total_flops / (path_b_median_busy_us * 1e-6)) / 1e12;

    let path_a_tokens_per_sec = (num_tokens as f64) / (path_a_median_busy_us * 1e-6);
    let path_b_tokens_per_sec = (num_tokens as f64) / (path_b_median_busy_us * 1e-6);

    println!("[4/4] Comparative performance metrics:");
    println!(
        "\nResults summary ({} iters, per dispatch):",
        cfg.bench_iters
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:<16} | {:<16} | {:<14}",
        "Metric", "Charon Resident", "Scythe Dynamic", "Speedup"
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:>13.2} us | {:>13.2} us | {:>12.2}x ({})",
        "Kernel Busy Time (Sync Median)",
        path_a_median_busy_us,
        path_b_median_busy_us,
        path_a_median_busy_us / path_b_median_busy_us,
        if path_b_median_busy_us < path_a_median_busy_us {
            "Scythe faster"
        } else {
            "Charon faster"
        }
    );
    println!(
        "{:<36} | {:>13.2} us | {:>13.2} us | {:>12.2}x ({})",
        "Pipelined Time (Batched+Sync)",
        path_a_total_wall_us,
        path_b_total_wall_us,
        path_a_total_wall_us / path_b_total_wall_us,
        if path_b_total_wall_us < path_a_total_wall_us {
            "Scythe faster"
        } else {
            "Charon faster"
        }
    );
    println!(
        "{:<36} | {:>13.2} us | {:>13.2} us | {:>12.2}x ({})",
        "Host Dispatch Wall Overhead",
        path_a_launch_wall_us,
        path_b_launch_wall_us,
        path_a_launch_wall_us / path_b_launch_wall_us,
        if path_b_launch_wall_us < path_a_launch_wall_us {
            "Scythe faster"
        } else {
            "Charon faster"
        }
    );
    println!(
        "{:<36} | {:>13.4} T | {:>13.4} T | {:>12.2}x",
        "Throughput (Compute TFLOPS)",
        path_a_tflops,
        path_b_tflops,
        path_b_tflops / path_a_tflops
    );
    println!(
        "{:<36} | {:>13.0}   | {:>13.0}   | {:>12.2}x",
        "Throughput (Tokens / sec)",
        path_a_tokens_per_sec,
        path_b_tokens_per_sec,
        path_b_tokens_per_sec / path_a_tokens_per_sec
    );
    println!("-----------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_moe_dispatch_scythe_ab() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU MoE A/B benchmark (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Config 1: Production Decode Shape (tokens=1, hidden=2048, inter=512, experts=8, top_k=2)
    run_ab_benchmark_config(&BenchConfig {
        name: "Full Decode Shape (2048/512)",
        num_tokens: 1,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 2,
        warmup_iters: 5,
        bench_iters: 20,
    });

    // Config 2: Batched Decode Shape (tokens=4, hidden=2048, inter=512, experts=8, top_k=2)
    run_ab_benchmark_config(&BenchConfig {
        name: "Batched Decode Shape (2048/512)",
        num_tokens: 4,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 8,
        warmup_iters: 5,
        bench_iters: 20,
    });
}
