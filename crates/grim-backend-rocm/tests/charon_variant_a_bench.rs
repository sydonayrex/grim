//! Benchmark Charon Baseline vs Variant A (CTA Parallelism):
//!   - Baseline: `moe_fused_dispatch_resident_routing_into` (1 thread per pair)
//!   - Variant A: `moe_fused_dispatch_resident_routing_into_cta` (256 threads per pair)
//!
//! Run with:
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm --test charon_variant_a_bench -- --nocapture

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

#[test]
fn bench_charon_baseline_vs_variant_a() {
    if !gpu_test_enabled() {
        eprintln!("Skipping GPU test (GRIM_RUN_GPU_TEST not set)");
        return;
    }

    let _lock = gpu_test_lock();
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new");

    let batch = 1usize;
    let hidden = 2048usize;
    let inter = 512usize;
    let num_experts = 8usize;
    let top_k = 2usize;
    let num_pairs = batch * top_k;
    let warmup_iters = 5;
    let bench_iters = 20;

    println!("\n=========================================================================================");
    println!("Charon Baseline vs Variant A (CTA Parallelism): batch={}, hidden={}, inter={}, pairs={}", batch, hidden, inter, num_pairs);
    println!("=========================================================================================");

    let activations: Vec<f32> = (0..(batch * hidden)).map(|i| ((i % 100) as f32 + 1.0) * 0.01).collect();
    let expert_gate_w: Vec<f32> = (0..(num_experts * inter * hidden)).map(|i| (((i % 13) as f32) - 6.0) * 0.005).collect();
    let expert_up_w: Vec<f32> = (0..(num_experts * inter * hidden)).map(|i| (((i % 11) as f32) - 5.0) * 0.005).collect();
    let expert_down_w: Vec<f32> = (0..(num_experts * hidden * inter)).map(|i| (((i % 7) as f32) - 3.0) * 0.005).collect();

    let token_ids: Vec<u32> = vec![0, 0];
    let expert_ids: Vec<u32> = vec![0, 1];
    let weights: Vec<f32> = vec![0.5, 0.5];

    let act_storage = upload_f32(&dev, &activations, Shape::new(vec![batch, hidden]));
    let gw_storage = upload_f32(&dev, &expert_gate_w, Shape::new(vec![expert_gate_w.len()]));
    let uw_storage = upload_f32(&dev, &expert_up_w, Shape::new(vec![expert_up_w.len()]));
    let dw_storage = upload_f32(&dev, &expert_down_w, Shape::new(vec![expert_down_w.len()]));

    let ti_storage = upload_u32(&dev, &token_ids, Shape::new(vec![num_pairs]));
    let ei_storage = upload_u32(&dev, &expert_ids, Shape::new(vec![num_pairs]));
    let w_storage = upload_f32(&dev, &weights, Shape::new(vec![num_pairs]));

    let out_base_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));
    let out_a_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));
    let out_ab_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));
    let out_abc_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));
    let out_nca_storage = upload_f32(&dev, &vec![0.0f32; batch * hidden], Shape::new(vec![batch, hidden]));

    let act_s = act_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ti_s = ti_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let ei_s = ei_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let w_s = w_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_base_s = out_base_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_a_s = out_a_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_ab_s = out_ab_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_abc_s = out_abc_storage.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_nca_s = out_nca_storage.as_any().downcast_ref::<RocmStorage>().unwrap();

    // 1. Parity Check
    print!("[1/6] Parity check Baseline vs Variants A, A+B, A+B+C, Normal+C+A... ");
    dev.moe_fused_dispatch_resident_routing_into(
        act_s,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_s,
        ei_s,
        w_s,
        num_pairs,
        out_base_s,
        hidden,
        inter,
        1.0,
    ).unwrap();

    dev.moe_fused_dispatch_resident_routing_into_cta(
        act_s,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_s,
        ei_s,
        w_s,
        num_pairs,
        out_a_s,
        hidden,
        inter,
        1.0,
    ).unwrap();

    dev.moe_fused_dispatch_resident_routing_into_ab(
        act_s,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_s,
        ei_s,
        w_s,
        num_pairs,
        out_ab_s,
        hidden,
        inter,
        1.0,
    ).unwrap();

    dev.moe_fused_dispatch_resident_routing_into_abc(
        act_s,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_s,
        ei_s,
        w_s,
        num_pairs,
        out_abc_s,
        hidden,
        inter,
        1.0,
    ).unwrap();

    dev.moe_fused_dispatch_resident_routing_into_normal_ca(
        act_s,
        gw_storage.as_ref(),
        uw_storage.as_ref(),
        dw_storage.as_ref(),
        ti_s,
        ei_s,
        w_s,
        num_pairs,
        out_nca_s,
        hidden,
        inter,
        1.0,
    ).unwrap();

    dev.synchronize();

    let r_base = out_base_s.to_cpu_vec_f32().unwrap();
    let r_a = out_a_s.to_cpu_vec_f32().unwrap();
    let r_ab = out_ab_s.to_cpu_vec_f32().unwrap();
    let r_abc = out_abc_s.to_cpu_vec_f32().unwrap();
    let r_nca = out_nca_s.to_cpu_vec_f32().unwrap();

    let mut max_diff_a = 0.0f32;
    let mut max_diff_ab = 0.0f32;
    let mut max_diff_abc = 0.0f32;
    let mut max_diff_nca = 0.0f32;
    for i in 0..r_base.len() {
        let da = (r_base[i] - r_a[i]).abs();
        if da > max_diff_a { max_diff_a = da; }
        let dab = (r_base[i] - r_ab[i]).abs();
        if dab > max_diff_ab { max_diff_ab = dab; }
        let dabc = (r_base[i] - r_abc[i]).abs();
        if dabc > max_diff_abc { max_diff_abc = dabc; }
        let dnca = (r_base[i] - r_nca[i]).abs();
        if dnca > max_diff_nca { max_diff_nca = dnca; }
    }
    println!("OK (diffs: A={:.2e}, A+B={:.2e}, A+B+C={:.2e}, Normal+C+A={:.2e})", max_diff_a, max_diff_ab, max_diff_abc, max_diff_nca);
    assert!(max_diff_a < 1e-4, "Parity check failed: max diff A {:.6e} >= 1e-4", max_diff_a);
    assert!(max_diff_ab < 1e-4, "Parity check failed: max diff A+B {:.6e} >= 1e-4", max_diff_ab);
    assert!(max_diff_abc < 1e-4, "Parity check failed: max diff A+B+C {:.6e} >= 1e-4", max_diff_abc);
    assert!(max_diff_nca < 1e-4, "Parity check failed: max diff Normal+C+A {:.6e} >= 1e-4", max_diff_nca);

    // 2. Warmup & Benchmark Baseline
    for _ in 0..warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_base_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
    }
    dev.synchronize();

    print!("[2/6] Timing Baseline Charon Resident MoE... ");
    let mut times_base: Vec<f64> = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
        let t0 = Instant::now();
        dev.moe_fused_dispatch_resident_routing_into(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_base_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
        dev.synchronize();
        times_base.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 3. Warmup & Benchmark Variant A
    for _ in 0..warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into_cta(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_a_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
    }
    dev.synchronize();

    print!("[3/6] Timing Variant A (CTA-parallel)... ");
    let mut times_a: Vec<f64> = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
        let t0 = Instant::now();
        dev.moe_fused_dispatch_resident_routing_into_cta(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_a_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
        dev.synchronize();
        times_a.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 4. Warmup & Benchmark Variant A+B
    for _ in 0..warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into_ab(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_ab_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
    }
    dev.synchronize();

    print!("[4/6] Timing Variant A+B (CTA + Local Reduction)... ");
    let mut times_ab: Vec<f64> = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
        let t0 = Instant::now();
        dev.moe_fused_dispatch_resident_routing_into_ab(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_ab_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
        dev.synchronize();
        times_ab.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 5. Warmup & Benchmark Variant A+B+C
    for _ in 0..warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into_abc(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_abc_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
    }
    dev.synchronize();

    print!("[5/6] Timing Variant A+B+C (A+B + float4 loads)... ");
    let mut times_abc: Vec<f64> = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
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
            out_abc_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
        dev.synchronize();
        times_abc.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    // 6. Warmup & Benchmark Variant Normal+C+A
    for _ in 0..warmup_iters {
        dev.moe_fused_dispatch_resident_routing_into_normal_ca(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_nca_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
    }
    dev.synchronize();

    print!("[6/6] Timing Variant Normal+C+A (CTA + float4 loads + immediate down)... ");
    let mut times_nca: Vec<f64> = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
        let t0 = Instant::now();
        dev.moe_fused_dispatch_resident_routing_into_normal_ca(
            act_s,
            gw_storage.as_ref(),
            uw_storage.as_ref(),
            dw_storage.as_ref(),
            ti_s,
            ei_s,
            w_s,
            num_pairs,
            out_nca_s,
            hidden,
            inter,
            1.0,
        ).unwrap();
        dev.synchronize();
        times_nca.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("DONE");

    times_base.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_ab.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_abc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_nca.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med_base = times_base[bench_iters / 2];
    let med_a = times_a[bench_iters / 2];
    let med_ab = times_ab[bench_iters / 2];
    let med_abc = times_abc[bench_iters / 2];
    let med_nca = times_nca[bench_iters / 2];

    println!("\nFinal Comparative Summary across all Optimization Variants:");
    println!("============================================================================================================================");
    println!("{:<32} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}", "Metric", "Baseline", "Variant A", "Variant A+B", "Variant A+B+C", "Normal+C+A");
    println!("============================================================================================================================");
    println!("{:<32} | {:>13.2} us | {:>13.2} us | {:>13.2} us | {:>13.2} us | {:>13.2} us", "Latency (Sync Median)", med_base, med_a, med_ab, med_abc, med_nca);
    println!("{:<32} | {:>16} | {:>13.2}x | {:>13.2}x | {:>13.2}x | {:>13.2}x", "Speedup vs Baseline", "1.00x", med_base / med_a, med_base / med_ab, med_base / med_abc, med_base / med_nca);
    println!("{:<32} | {:>16} | {:>16} | {:>13.2}x | {:>13.2}x | {:>13.2}x", "Speedup vs Variant A", "-", "1.00x", med_a / med_ab, med_a / med_abc, med_a / med_nca);
    println!("============================================================================================================================\n");
}
