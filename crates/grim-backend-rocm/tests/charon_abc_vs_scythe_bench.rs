//! Microbenchmark comparing Optimized Charon Resident MoE (Variant A+B+C) vs Scythe WMMA GEMM:
//!   - Path A: Charon Resident MoE Optimized Variant A+B+C (`moe_fused_dispatch_resident_routing_into_abc`)
//!   - Path B: Scythe WMMA GEMM (`launch_charon_grouped_dispatch_entry` with `grim_moe_fused_grouped_wmma`)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test charon_abc_vs_scythe_bench -- --nocapture

use std::time::Instant;

use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::kernels::charon::{moe_align_block_size, RoutingAssignment};
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

fn run_charon_abc_vs_scythe_bench(cfg: &BenchConfig) {
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
        "Charon Resident (Variant A+B+C) vs Scythe WMMA GEMM [{}]: batch={}, hidden={}, inter={}, experts={}, top_k={}",
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

    let act_storage = upload_f32(&dev, &activations, Shape::new(vec![batch, hidden]));
    let gw_storage = upload_f32(&dev, &expert_gate_w, Shape::new(vec![expert_gate_w.len()]));
    let uw_storage = upload_f32(&dev, &expert_up_w, Shape::new(vec![expert_up_w.len()]));
    let dw_storage = upload_f32(&dev, &expert_down_w, Shape::new(vec![expert_down_w.len()]));

    let ti_storage = upload_u32(&dev, &token_ids, Shape::new(vec![num_pairs]));
    let ei_storage = upload_u32(&dev, &expert_ids, Shape::new(vec![num_pairs]));
    let w_storage = upload_f32(&dev, &weights, Shape::new(vec![num_pairs]));

    let out_charon_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));
    let out_scythe_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let gw_s = gw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let uw_s = uw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let dw_s = dw_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ti_s = ti_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ei_s = ei_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let w_s = w_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_charon_s = out_charon_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_scythe_s = out_scythe_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    let gw_ptr = dev_ptr(gw_s).expect("gw_ptr");
    let uw_ptr = dev_ptr(uw_s).expect("uw_ptr");
    let dw_ptr = dev_ptr(dw_s).expect("dw_ptr");

    // 1. Parity Check
    print!("[1/4] Parity check Charon Resident (A+B+C) vs Scythe WMMA GEMM... ");
    dev.moe_fused_dispatch_resident_routing_into_abc(
        act_s,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_s,
        ei_s,
        w_s,
        num_pairs,
        out_charon_s,
        hidden,
        inter,
        1.0,
    )
    .expect("Charon launch");

    let _ = dev
        .launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_scythe_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        )
        .expect("Scythe WMMA launch");

    dev.synchronize();

    let res_charon = out_charon_s.to_cpu_vec_f32().unwrap();
    let res_scythe = out_scythe_s.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for (v_c, v_s) in res_charon.iter().zip(res_scythe.iter()) {
        let diff = (v_c - v_s).abs();
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

    // 2. Warmup & Benchmark Charon Resident (A+B+C)
    for _ in 0..cfg.warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into_abc(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_charon_s,
            hidden,
            inter,
            1.0,
        )
        .unwrap();
    }
    dev.synchronize();

    print!("[2/4] Benchmarking Charon Resident (Variant A+B+C)... ");
    let mut charon_sync_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        dev.moe_fused_dispatch_resident_routing_into_abc(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_charon_s,
            hidden,
            inter,
            1.0,
        )
        .unwrap();
        dev.synchronize();
        charon_sync_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Warmup & Benchmark Scythe WMMA GEMM
    for _ in 0..cfg.warmup_iters {
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_scythe_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        )
        .unwrap();
    }
    dev.synchronize();

    print!("[3/4] Benchmarking Scythe WMMA GEMM... ");
    let mut scythe_sync_times_us: Vec<f64> = Vec::with_capacity(cfg.bench_iters);
    for _ in 0..cfg.bench_iters {
        let t0 = Instant::now();
        let _ = dev.launch_charon_grouped_dispatch_entry(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_scythe_s,
            hidden,
            inter,
            1.0,
            num_experts,
            "grim_moe_fused_grouped_wmma",
            None,
            None,
        )
        .unwrap();
        dev.synchronize();
        scythe_sync_times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    charon_sync_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    scythe_sync_times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let charon_median_us = charon_sync_times_us[cfg.bench_iters / 2];
    let scythe_median_us = scythe_sync_times_us[cfg.bench_iters / 2];

    let flops_per_token = (3 * 2 * hidden * inter) as f64;
    let total_flops = flops_per_token * (batch as f64) * (top_k as f64);

    let charon_tflops = (total_flops / (charon_median_us * 1e-6)) / 1e12;
    let scythe_tflops = (total_flops / (scythe_median_us * 1e-6)) / 1e12;

    let charon_tokens_per_sec = (batch as f64) / (charon_median_us * 1e-6);
    let scythe_tokens_per_sec = (batch as f64) / (scythe_median_us * 1e-6);

    println!("[4/4] Comparative Summary (Iterations = {}):", cfg.bench_iters);
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:<20} | {:<16} | {:<14}",
        "Metric", "Charon (Var A+B+C)", "Scythe WMMA GEMM", "Speedup"
    );
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<36} | {:>17.2} us | {:>13.2} us | {:>12.2}x ({})",
        "Kernel Latency (Sync Median)",
        charon_median_us,
        scythe_median_us,
        charon_median_us / scythe_median_us,
        if scythe_median_us < charon_median_us {
            "Scythe WMMA faster"
        } else {
            "Charon faster"
        }
    );
    println!(
        "{:<36} | {:>17.4} T | {:>13.4} T | {:>12.2}x",
        "Throughput (Compute TFLOPS)",
        charon_tflops,
        scythe_tflops,
        scythe_tflops / charon_tflops
    );
    println!(
        "{:<36} | {:>17.0}   | {:>13.0}   | {:>12.2}x",
        "Throughput (Tokens / sec)",
        charon_tokens_per_sec,
        scythe_tokens_per_sec,
        scythe_tokens_per_sec / charon_tokens_per_sec
    );
    println!("-----------------------------------------------------------------------------------------\n");
}

#[test]
fn bench_charon_abc_vs_scythe_gemm() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();

    // Config 1: Production Decode Shape (batch=1, hidden=2048, inter=512, experts=8, top_k=2)
    run_charon_abc_vs_scythe_bench(&BenchConfig {
        name: "Single Decode (M=1)",
        batch: 1,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 2,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Config 2: Batched Decode Shape (batch=4, hidden=2048, inter=512, experts=8, top_k=2)
    run_charon_abc_vs_scythe_bench(&BenchConfig {
        name: "Batched Decode (M=4)",
        batch: 4,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 8,
        warmup_iters: 10,
        bench_iters: 40,
    });

    // Config 3: Prefill Tile Shape (batch=16, hidden=2048, inter=512, experts=8, top_k=2)
    run_charon_abc_vs_scythe_bench(&BenchConfig {
        name: "Prefill Tile (M=16)",
        batch: 16,
        hidden: 2048,
        inter: 512,
        num_experts: 8,
        top_k: 2,
        block_size: 16,
        warmup_iters: 10,
        bench_iters: 40,
    });
}
