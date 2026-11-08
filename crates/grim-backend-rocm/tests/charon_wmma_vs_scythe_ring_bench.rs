//! Microbenchmark comparing:
//!   - Path A: Charon WMMA (`grim_moe_fused_grouped_wmma` launched on compute stream)
//!   - Path B: Scythe Ring MoE (`enqueue_dynamic_moe` executed via Scythe persistent ring wave)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test charon_wmma_vs_scythe_ring_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::kernels::charon::{moe_align_block_size, RoutingAssignment};
use grim_backend_rocm::kernels::task_ring::TaskRingContext;
use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
use grim_backend_rocm::{dev_ptr, gpu_test_enabled, RocmDevice, RocmStorage};
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
    batch: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    block_size: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

fn run_charon_wmma_vs_scythe_ring(cfg: &BenchConfig) {
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
        "Charon WMMA vs Scythe Ring MoE [{}]: batch={}, hidden={}, inter={}, experts={}, top_k={}",
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
        tokens: token_ids.clone(),
        experts: expert_ids.clone(),
        weights: weights.clone(),
    };

    let sorted = moe_align_block_size(&assignment, block_size, num_experts);

    let act_shape = Shape::new(vec![batch, hidden]);
    let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
    let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
    let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
    let out_shape = Shape::new(vec![batch, hidden]);

    let act_storage = upload_f32(&dev, &activations, act_shape.clone());
    let gw_storage = upload_f32(&dev, &expert_gate_w, exp_gate_shape);
    let uw_storage = upload_f32(&dev, &expert_up_w, exp_up_shape);
    let dw_storage = upload_f32(&dev, &expert_down_w, exp_down_shape);

    let ti_storage = upload_u32(&dev, &token_ids, Shape::new(vec![num_pairs]));
    let ei_storage = upload_u32(&dev, &expert_ids, Shape::new(vec![num_pairs]));
    let w_storage = upload_f32(&dev, &weights, Shape::new(vec![num_pairs]));

    let out_wmma_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], out_shape.clone());
    let out_ring_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], out_shape.clone());

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let gw_s = gw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let uw_s = uw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let dw_s = dw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ti_s = ti_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ei_s = ei_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let w_s = w_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_wmma_s = out_wmma_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_ring_s = out_ring_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    let gw_ptr = dev_ptr(gw_s).unwrap();
    let uw_ptr = dev_ptr(uw_s).unwrap();
    let dw_ptr = dev_ptr(dw_s).unwrap();
    let act_ptr = dev_ptr(act_s).unwrap();
    let ti_ptr = dev_ptr(ti_s).unwrap();
    let ei_ptr = dev_ptr(ei_s).unwrap();
    let w_ptr = dev_ptr(w_s).unwrap();
    let out_ring_ptr = dev_ptr(out_ring_s).unwrap();

    // Prepare TaskRingContext for Scythe Ring MoE (Path B)
    let mut graph_b = TaskGraphBuilder::new();
    let h_in = graph_b.push_arg(act_ptr, (batch * hidden * 4) as u64, arg_role::ACT_IN);
    let h_out = graph_b.push_arg(out_ring_ptr, (batch * hidden * 4) as u64, arg_role::ACT_OUT);
    let h_gw = graph_b.push_arg(gw_ptr, (expert_gate_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_uw = graph_b.push_arg(uw_ptr, (expert_up_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_dw = graph_b.push_arg(dw_ptr, (expert_down_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_ti = graph_b.push_arg(ti_ptr, (num_pairs * 4) as u64, arg_role::SCRATCH);
    let h_ei = graph_b.push_arg(ei_ptr, (num_pairs * 4) as u64, arg_role::SCRATCH);
    let h_w = graph_b.push_arg(w_ptr, (num_pairs * 4) as u64, arg_role::SCRATCH);

    let params = [
        0u32, // quant_mode = 0 (FP32)
        hidden as u32,
        inter as u32,
        batch as u32,
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

    // 1. Parity Check
    print!("[1/4] Numerical parity check (Charon WMMA vs Scythe Ring MoE)... ");
    let _ = dev
        .launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_wmma_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        )
        .expect("WMMA launch");

    ring_ctx
        .enqueue_dynamic_moe(&dev, moe_binding)
        .expect("Scythe Ring launch");

    dev.synchronize();

    let res_wmma = out_wmma_s.to_cpu_vec_f32().unwrap();
    let res_ring = out_ring_s.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for (v_w, v_r) in res_wmma.iter().zip(res_ring.iter()) {
        let diff = (v_w - v_r).abs();
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

    // 2. Warmup & Benchmark Path A: Charon WMMA
    for _ in 0..cfg.warmup_iters {
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_wmma_s,
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

    print!("[2/4] Benchmarking Path A: Charon WMMA... ");
    let mut wmma_sync_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_wmma_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        );
        dev.synchronize();
        wmma_sync_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Warmup & Benchmark Path B: Scythe Ring MoE
    for _ in 0..cfg.warmup_iters {
        ring_ctx.enqueue_dynamic_moe(&dev, moe_binding).unwrap();
    }
    dev.synchronize();

    print!("[3/4] Benchmarking Path B: Scythe Ring MoE... ");
    let mut ring_sync_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        ring_ctx.enqueue_dynamic_moe(&dev, moe_binding).unwrap();
        dev.synchronize();
        ring_sync_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    wmma_sync_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ring_sync_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let wmma_median_us = wmma_sync_times_us[cfg.bench_iters / 2];
    let ring_median_us = ring_sync_times_us[cfg.bench_iters / 2];

    let total_flops = (num_pairs as f64) * (6.0 * hidden as f64 * inter as f64);
    let wmma_tflops = (total_flops / (wmma_median_us * 1e-6)) / 1e12;
    let ring_tflops = (total_flops / (ring_median_us * 1e-6)) / 1e12;

    let wmma_tokens_per_sec = (batch as f64) / (wmma_median_us * 1e-6);
    let ring_tokens_per_sec = (batch as f64) / (ring_median_us * 1e-6);

    println!("[4/4] Comparative Performance Summary:");
    println!(
        "\nResults summary ({} iters, per dispatch):",
        cfg.bench_iters
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:<16} | {:<16} | {:<14}",
        "Metric", "Charon WMMA", "Scythe Ring MoE", "Speedup"
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:>13.2} us | {:>13.2} us | {:>12.2}x ({})",
        "Kernel Latency (Sync Median)",
        wmma_median_us,
        ring_median_us,
        ring_median_us / wmma_median_us,
        if wmma_median_us < ring_median_us {
            "Charon WMMA faster"
        } else {
            "Scythe Ring faster"
        }
    );
    println!(
        "{:<36} | {:>13.4} T | {:>13.4} T | {:>12.2}x",
        "Throughput (Compute TFLOPS)",
        wmma_tflops,
        ring_tflops,
        wmma_tflops / ring_tflops
    );
    println!(
        "{:<36} | {:>13.0}   | {:>13.0}   | {:>12.2}x",
        "Throughput (Tokens / sec)",
        wmma_tokens_per_sec,
        ring_tokens_per_sec,
        wmma_tokens_per_sec / ring_tokens_per_sec
    );
    println!("-----------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_charon_wmma_vs_scythe_ring() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Config 1: Production Decode Shape (batch=1, hidden=2048, inter=512, experts=8, top_k=2)
    run_charon_wmma_vs_scythe_ring(&BenchConfig {
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
    run_charon_wmma_vs_scythe_ring(&BenchConfig {
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
    run_charon_wmma_vs_scythe_ring(&BenchConfig {
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
