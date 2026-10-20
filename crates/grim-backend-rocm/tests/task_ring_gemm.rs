//! Plan Phase M task M4: the persistent task ring — op::GEMM executed by ONE
//! launch for the whole step.
//!
//! Two gates here, deliberately split:
//!
//! 1. `the_plan_binds_only_what_a_ring_can_run` is CPU-only and pins the
//!    BINDING RULES (a node the ring cannot execute must be absent from the
//!    plan, not half-bound) plus the staging layout. This is the half that can
//!    be checked without a GPU and without a model.
//!
//! 2. `one_wave_launch_runs_the_steps_gemms` builds real device buffers, stages
//!    a plan, and asserts the two things M4 exists for: the GEMMs compute the
//!    right answer, and the whole set costs ONE launch (`wave_launch_count`
//!    advances by exactly one, however many GEMMs the step has).
//!
//! The third gate — the M2 oracle with the ring SUBSTITUTED for the GEMM
//! dispatch — lives with the oracle, because it needs a model's IR. See
//! `task_graph_parity` (`GRIM_TASK_RING=1`).

use grim_backend_rocm::device::util::DeviceGuard;
use grim_backend_rocm::kernels::task_ring::{
    TaskRingGemm, TaskRingPlan, STAGED_CAPACITY, launch_task_ring, wave_launch_count,
};
use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

/// M5: the plan builder's binding rules for the new opcodes.
///
/// ELEMENTWISE binds as OP_ADD (two inputs + output), ATTENTION binds as
/// OP_ATTN (q/k/v/out + geometry, head_dim ≤ 256), MOE_DISPATCH binds as
/// OP_MOE (all descriptor fields resolved). A node that fails its check is
/// absent from the plan and keeps running on its ordinary launcher.
#[test]
fn the_plan_binds_new_opcodes() {
    let a_ptr = 0x1000u64;
    let b_ptr = 0x2000u64;
    let o_ptr = 0x3000u64;
    let q_ptr = 0x4000u64;
    let k_ptr = 0x5000u64;
    let v_ptr = 0x6000u64;
    let ao_ptr = 0x7000u64;
    let gate_ptr = 0x8000u64;
    let up_ptr = 0x9000u64;
    let down_ptr = 0xA000u64;
    let ti_ptr = 0xB000u64;
    let ei_ptr = 0xC000u64;
    let w_ptr = 0xD000u64;

    let mut b = TaskGraphBuilder::new();

    // ELEMENTWISE: two ACT_IN operands + output. params[0]=m, params[1]=n.
    let h_a = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_b = b.push_arg(b_ptr, 64, arg_role::ACT_IN);
    let h_o = b.push_arg(o_ptr, 64, arg_role::ACT_OUT);
    b.push(op::ELEMENTWISE, [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_b, h_o], &[], 0, 0);

    // ATTENTION: q (ACT_IN), k (WEIGHT), v (SCRATCH), out (ACT_OUT).
    // params[0]=seq_len, params[1]=num_heads, params[2]=head_dim.
    let h_q = b.push_arg(q_ptr, 64, arg_role::ACT_IN);
    let h_k = b.push_arg(k_ptr, 64, arg_role::WEIGHT);
    let h_v = b.push_arg(v_ptr, 64, arg_role::SCRATCH);
    let h_ao = b.push_arg(ao_ptr, 64, arg_role::ACT_OUT);
    b.push(op::ATTENTION, [1, 2, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_q, h_k, h_v, h_ao], &[], 0, 0);

    // MOE_DISPATCH: input (ACT_IN), output (ACT_OUT), three WEIGHT args
    // (gate/up/down), three SCRATCH args (token_ids/expert_ids/weights).
    // params[0]=quant_mode, [1]=hidden, [2]=inter, [3]=num_tokens,
    // [4]=block_size, [5]=num_experts, [6]=top_k.
    let h_in = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_out = b.push_arg(o_ptr, 64, arg_role::ACT_OUT);
    let h_gw = b.push_arg(gate_ptr, 64, arg_role::WEIGHT);
    let h_uw = b.push_arg(up_ptr, 64, arg_role::WEIGHT);
    let h_dw = b.push_arg(down_ptr, 64, arg_role::WEIGHT);
    let h_ti = b.push_arg(ti_ptr, 64, arg_role::SCRATCH);
    let h_ei = b.push_arg(ei_ptr, 64, arg_role::SCRATCH);
    let h_w = b.push_arg(w_ptr, 64, arg_role::SCRATCH);
    b.push(op::MOE_DISPATCH, [0, 64, 128, 1, 32, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_in, h_out, h_gw, h_uw, h_dw, h_ti, h_ei, h_w], &[], 0, 0);

    // A quantized MoE (quant_mode != 0) must NOT bind.
    let h_in2 = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_out2 = b.push_arg(o_ptr, 64, arg_role::ACT_OUT);
    let h_gw2 = b.push_arg(gate_ptr, 64, arg_role::WEIGHT);
    let h_uw2 = b.push_arg(up_ptr, 64, arg_role::WEIGHT);
    let h_dw2 = b.push_arg(down_ptr, 64, arg_role::WEIGHT);
    let h_ti2 = b.push_arg(ti_ptr, 64, arg_role::SCRATCH);
    let h_ei2 = b.push_arg(ei_ptr, 64, arg_role::SCRATCH);
    let h_w2 = b.push_arg(w_ptr, 64, arg_role::SCRATCH);
    b.push(op::MOE_DISPATCH, [4, 64, 128, 1, 32, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_in2, h_out2, h_gw2, h_uw2, h_dw2, h_ti2, h_ei2, h_w2], &[], 0, 0);

    // An ATTENTION with head_dim > 256 must NOT bind.
    let h_q2 = b.push_arg(q_ptr, 64, arg_role::ACT_IN);
    let h_k2 = b.push_arg(k_ptr, 64, arg_role::WEIGHT);
    let h_v2 = b.push_arg(v_ptr, 64, arg_role::SCRATCH);
    let h_ao2 = b.push_arg(ao_ptr, 64, arg_role::ACT_OUT);
    b.push(op::ATTENTION, [1, 2, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_q2, h_k2, h_v2, h_ao2], &[], 0, 0);

    let g = b.finish();
    let plan = TaskRingPlan::from_graph(&g);

    assert_eq!(plan.adds.len(), 1, "one ELEMENTWISE node binds");
    assert_eq!(plan.attns.len(), 1, "one ATTENTION node binds (head_dim 64)");
    assert_eq!(plan.moes.len(), 1, "one MOE_DISPATCH node binds (FP32)");
    assert_eq!(plan.gemms.len(), 0, "no GEMM nodes");
    assert_eq!(plan.norms.len(), 0, "no NORM nodes");

    // The quantized MoE and the head_dim>256 attention are absent.
    assert_eq!(plan.step_len(), 3, "three executable nodes total");
}

/// M5: the ring must produce byte-identical results across two runs of the
/// same computation. This is the determinism gate the plan says "grim paid
/// for this lesson once already" — a device-side scheduler makes the ordering
/// question worse, not better, because the schedule is now chosen at runtime.
#[test]
fn ring_produces_byte_identical_results_across_runs() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // Two element-wise adds over the same data, run twice.
    let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32 * 0.25).collect();
    let (_a_s, a_ptr) = f32_dev(&dev, &a);
    let (_b_s, b_ptr) = f32_dev(&dev, &b);
    let (o1_s, o1_ptr) = f32_dev(&dev, &[0.0f32; 64]);
    let (o2_s, o2_ptr) = f32_dev(&dev, &[0.0f32; 64]);

    // Build a task graph with two ELEMENTWISE nodes and create a ring context.
    let mut graph_b = TaskGraphBuilder::new();
    let h_a = graph_b.push_arg(a_ptr, 256, arg_role::ACT_IN);
    let h_b = graph_b.push_arg(b_ptr, 256, arg_role::ACT_IN);
    let h_o1 = graph_b.push_arg(o1_ptr, 256, arg_role::ACT_OUT);
    let h_o2 = graph_b.push_arg(o2_ptr, 256, arg_role::ACT_OUT);
    graph_b.push(op::ELEMENTWISE, [1, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_b, h_o1], &[], 0, 0);
    graph_b.push(op::ELEMENTWISE, [1, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_b, h_o2], &[], 0, 0);
    let graph = graph_b.finish();

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds the adds");

    // Run twice.
    ctx.enqueue_add(&dev, &ctx.plan().adds()[0]).expect("enqueue add 1");
    ctx.enqueue_add(&dev, &ctx.plan().adds()[1]).expect("enqueue add 2");
    dev.synchronize();

    // Read back both outputs and assert byte-identical.
    let r1: Vec<f32> = o1_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("o1 is RocmStorage")
        .to_cpu_vec_f32()
        .expect("o1 readback");
    let r2: Vec<f32> = o2_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("o2 is RocmStorage")
        .to_cpu_vec_f32()
        .expect("o2 readback");
    assert_eq!(r1, r2, "two runs of the same add must be byte-identical");

    // Also verify the result is correct.
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((r1[i] - (x + y)).abs() < 1e-6, "add result wrong at {i}");
    }
}

/// M5 determinism gate: NORM must produce byte-identical results across two runs.
#[test]
fn ring_norm_produces_byte_identical_results_across_runs() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let a: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.25).collect();
    let w: Vec<f32> = (0..64).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let (_a_s, a_ptr) = f32_dev(&dev, &a);
    let (_w_s, w_ptr) = f32_dev(&dev, &w);
    let (o1_s, o1_ptr) = f32_dev(&dev, &[0.0f32; 64]);
    let (o2_s, o2_ptr) = f32_dev(&dev, &[0.0f32; 64]);

    let mut graph_b = TaskGraphBuilder::new();
    let h_a = graph_b.push_arg(a_ptr, 256, arg_role::ACT_IN);
    let h_w = graph_b.push_arg(w_ptr, 256, arg_role::WEIGHT);
    let h_o1 = graph_b.push_arg(o1_ptr, 256, arg_role::ACT_OUT);
    let h_o2 = graph_b.push_arg(o2_ptr, 256, arg_role::ACT_OUT);
    // params: [0, k=64, m=1, ...]
    graph_b.push(op::NORM, [0, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_w, h_o1], &[], 0, 0);
    graph_b.push(op::NORM, [0, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_w, h_o2], &[], 0, 0);
    let graph = graph_b.finish();

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds the norms");

    ctx.enqueue_norm(&dev, &ctx.norms()[0]).expect("enqueue norm 1");
    ctx.enqueue_norm(&dev, &ctx.norms()[1]).expect("enqueue norm 2");
    dev.synchronize();

    let r1: Vec<f32> = o1_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    let r2: Vec<f32> = o2_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    assert_eq!(r1, r2, "two runs of the same norm must be byte-identical");

    // ORACLE: out[k] = a[k] * rsqrt(mean(a^2) + eps) * w[k], eps = 1e-5 (the
    // wave arm's fixed constant). f64 on the host; `rsqrtf` is a device
    // approximation so the bar is 1e-4 relative, not exact.
    let ss: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let rms = 1.0 / ((ss / a.len() as f64) + 1e-5).sqrt();
    for j in 0..a.len() {
        let want = (a[j] as f64) * rms * (w[j] as f64);
        assert!(
            ((r1[j] as f64) - want).abs() <= 1e-4 * want.abs().max(1.0),
            "norm out[{j}] = {} but the host oracle says {want}",
            r1[j]
        );
    }
}

/// M5 determinism gate: GEMM must produce byte-identical results across two runs.
#[test]
fn ring_gemm_produces_byte_identical_results_across_runs() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let m = 1u32;
    let n = 4u32;
    let k = 8u32;
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.5).collect();
    let w: Vec<f32> = (0..(n * k)).map(|i| ((i % 5) as f32) * 0.25).collect();
    let (_a_s, a_ptr) = f32_dev(&dev, &a);
    let (_w_s, w_ptr) = f32_dev(&dev, &w);
    let (o1_s, o1_ptr) = f32_dev(&dev, &[0.0f32; 4]);
    let (o2_s, o2_ptr) = f32_dev(&dev, &[0.0f32; 4]);

    let mut graph_b = TaskGraphBuilder::new();
    let h_a = graph_b.push_arg(a_ptr, (k * 4) as u64, arg_role::ACT_IN);
    let h_w = graph_b.push_arg(w_ptr, (n * k * 4) as u64, arg_role::WEIGHT);
    let h_o1 = graph_b.push_arg(o1_ptr, (n * 4) as u64, arg_role::ACT_OUT);
    let h_o2 = graph_b.push_arg(o2_ptr, (n * 4) as u64, arg_role::ACT_OUT);
    // params: [quant=0, n=4, k=8, m=1, ...]
    graph_b.push(op::GEMM, [0, n, k, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_w, h_o1], &[], 0, 0);
    graph_b.push(op::GEMM, [0, n, k, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_a, h_w, h_o2], &[], 0, 0);
    let graph = graph_b.finish();

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds the gemms");

    ctx.enqueue_gemm(&dev, &ctx.plan().gemms[0]).expect("enqueue gemm 1");
    ctx.enqueue_gemm(&dev, &ctx.plan().gemms[1]).expect("enqueue gemm 2");
    dev.synchronize();

    let r1: Vec<f32> = o1_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    let r2: Vec<f32> = o2_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    assert_eq!(r1, r2, "two runs of the same gemm must be byte-identical");

    // ORACLE (ai-troubleshooting-expert: "a gate that cannot fail is not a
    // gate"). `r1 == r2` alone is satisfied by a ring that silently skips the
    // node — both outputs stay at their initial value. The VALUE is the real
    // check: C = A @ W^T, computed here in f64 on the host.
    for j in 0..n as usize {
        let mut want = 0.0f64;
        for i in 0..k as usize {
            want += (a[i] as f64) * (w[j * k as usize + i] as f64);
        }
        assert!(
            (r1[j] as f64 - want).abs() < 1e-5,
            "gemm out[{j}] = {} but the host oracle says {want}",
            r1[j]
        );
    }
}

/// M5 determinism gate: ATTENTION must produce byte-identical results across two runs.
#[test]
fn ring_attention_produces_byte_identical_results_across_runs() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let seq_len = 1u32;
    let num_heads = 2u32;
    let head_dim = 64u32;
    let total = (seq_len * num_heads * head_dim) as usize;

    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();
    let k: Vec<f32> = (0..total).map(|i| ((i + 1) as f32) * 0.03).collect();
    let v: Vec<f32> = (0..total).map(|i| ((i + 2) as f32) * 0.02).collect();

    let (_q_s, q_ptr) = f32_dev(&dev, &q);
    let (_k_s, k_ptr) = f32_dev(&dev, &k);
    let (_v_s, v_ptr) = f32_dev(&dev, &v);
    let (o1_s, o1_ptr) = f32_dev(&dev, &vec![0.0f32; total]);
    let (o2_s, o2_ptr) = f32_dev(&dev, &vec![0.0f32; total]);

    let mut graph_b = TaskGraphBuilder::new();
    let h_q = graph_b.push_arg(q_ptr, (total * 4) as u64, arg_role::ACT_IN);
    let h_k = graph_b.push_arg(k_ptr, (total * 4) as u64, arg_role::WEIGHT);
    let h_v = graph_b.push_arg(v_ptr, (total * 4) as u64, arg_role::SCRATCH);
    let h_o1 = graph_b.push_arg(o1_ptr, (total * 4) as u64, arg_role::ACT_OUT);
    let h_o2 = graph_b.push_arg(o2_ptr, (total * 4) as u64, arg_role::ACT_OUT);
    // params: [seq_len, num_heads, head_dim, ...]
    graph_b.push(op::ATTENTION, [seq_len, num_heads, head_dim, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_q, h_k, h_v, h_o1], &[], 0, 0);
    graph_b.push(op::ATTENTION, [seq_len, num_heads, head_dim, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[h_q, h_k, h_v, h_o2], &[], 0, 0);
    let graph = graph_b.finish();

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds the attns");

    ctx.enqueue_attn(&dev, &ctx.plan().attns()[0]).expect("enqueue attn 1");
    ctx.enqueue_attn(&dev, &ctx.plan().attns()[1]).expect("enqueue attn 2");
    dev.synchronize();

    let r1: Vec<f32> = o1_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    let r2: Vec<f32> = o2_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    assert_eq!(r1, r2, "two runs of the same attention must be byte-identical");

    // ORACLE, and an exact one: with seq_len = 1 the softmax is over a single
    // key, so it is 1 and out == V. No tolerance needed, and it pins the V
    // operand binding — a swapped k/v, or a node the ring skipped, fails here.
    assert_eq!(
        r1,
        v,
        "seq_len=1 attention must be exactly V; got {:?} vs {:?}",
        &r1[..4],
        &v[..4]
    );
}

/// M5 determinism gate: MOE_DISPATCH must produce byte-identical results across two runs.
#[test]
fn ring_moe_produces_byte_identical_results_across_runs() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let hidden = 8usize;
    let inter = 8usize;
    let num_experts = 4usize;
    let num_tokens = 2usize;
    let top_k = 2usize;
    let block_size = 1usize;

    let x: Vec<f32> = (0..(num_tokens * hidden)).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let gate_w: Vec<f32> = (0..(num_experts * inter * hidden)).map(|i| ((i % 7) as f32) * 0.05).collect();
    let up_w: Vec<f32> = (0..(num_experts * inter * hidden)).map(|i| ((i % 5) as f32) * 0.05).collect();
    let down_w: Vec<f32> = (0..(num_experts * hidden * inter)).map(|i| ((i % 3) as f32) * 0.05).collect();

    let token_ids: Vec<u32> = vec![0, 0, 1, 1];
    let expert_ids: Vec<u32> = vec![0, 1, 2, 3];
    let weights: Vec<f32> = vec![0.6, 0.4, 0.7, 0.3];

    let (_x_s, x_ptr) = f32_dev(&dev, &x);
    let (_gw_s, gw_ptr) = f32_dev(&dev, &gate_w);
    let (_uw_s, uw_ptr) = f32_dev(&dev, &up_w);
    let (_dw_s, dw_ptr) = f32_dev(&dev, &down_w);

    let (ti_s, _ti_ptr) = u32_dev(&dev, 0);
    let (ei_s, _ei_ptr) = u32_dev(&dev, 0);
    let (_w_s, w_ptr) = f32_dev(&dev, &weights);
    let _ = (ti_s, ei_s);

    // Upload token_ids & expert_ids vectors
    let ti_bytes: Vec<u8> = token_ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ti_storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &ti_bytes,
            &Shape::new(vec![token_ids.len()]),
            DType { arith: ArithType::U32, storage: DTypeStorage::Native },
        )
        .expect("token_ids upload");
    let ti_dev_ptr = ti_storage.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().device_ptr_u64().unwrap();

    let ei_bytes: Vec<u8> = expert_ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ei_storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &ei_bytes,
            &Shape::new(vec![expert_ids.len()]),
            DType { arith: ArithType::U32, storage: DTypeStorage::Native },
        )
        .expect("expert_ids upload");
    let ei_dev_ptr = ei_storage.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().device_ptr_u64().unwrap();

    let (o1_s, o1_ptr) = f32_dev(&dev, &vec![0.0f32; num_tokens * hidden]);
    let (o2_s, o2_ptr) = f32_dev(&dev, &vec![0.0f32; num_tokens * hidden]);

    let mut graph_b = TaskGraphBuilder::new();
    let h_in = graph_b.push_arg(x_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_IN);
    let h_out1 = graph_b.push_arg(o1_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_OUT);
    let h_out2 = graph_b.push_arg(o2_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_OUT);
    let h_gw = graph_b.push_arg(gw_ptr, (gate_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_uw = graph_b.push_arg(uw_ptr, (up_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_dw = graph_b.push_arg(dw_ptr, (down_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_ti = graph_b.push_arg(ti_dev_ptr, (token_ids.len() * 4) as u64, arg_role::SCRATCH);
    let h_ei = graph_b.push_arg(ei_dev_ptr, (expert_ids.len() * 4) as u64, arg_role::SCRATCH);
    let h_w = graph_b.push_arg(w_ptr, (weights.len() * 4) as u64, arg_role::SCRATCH);

    // params: [quant_mode=0, hidden, inter, num_tokens, block_size, num_experts, top_k]
    let params = [
        0u32, hidden as u32, inter as u32, num_tokens as u32, block_size as u32,
        num_experts as u32, top_k as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    graph_b.push(op::MOE_DISPATCH, params, &[h_in, h_out1, h_gw, h_uw, h_dw, h_ti, h_ei, h_w], &[], 0, 0);
    graph_b.push(op::MOE_DISPATCH, params, &[h_in, h_out2, h_gw, h_uw, h_dw, h_ti, h_ei, h_w], &[], 0, 0);
    let graph = graph_b.finish();

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds moes");

    ctx.enqueue_moe(&dev, &ctx.plan().moes()[0]).expect("enqueue moe 1");
    ctx.enqueue_moe(&dev, &ctx.plan().moes()[1]).expect("enqueue moe 2");
    dev.synchronize();

    let r1: Vec<f32> = o1_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    let r2: Vec<f32> = o2_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    assert_eq!(r1, r2, "two runs of the same moe dispatch must be byte-identical");

    // NON-VACUITY: `r1 == r2` is satisfied by a ring that skipped the node —
    // both outputs would sit at their initial value. The fused MoE kernel has no
    // host-callable launcher to diff against, so the cheap oracle here is that
    // the dispatch actually produced signal; its numerics are covered by the
    // kernel's own parity suite.
    assert!(
        r1.iter().any(|&x| x != 0.0),
        "moe dispatch produced all zeros — the ring never executed the node"
    );
}

/// M5 Step 3: Event-Tensor on-device dynamic successor enqueue test for MOE_DISPATCH.
///
/// Asserts that on-device dynamic enqueue (`enqueue_dynamic_moe`) produces
/// byte-identical results across two runs with zero host-staged descriptors.
#[test]
fn ring_dynamic_moe_produces_byte_identical_results_across_runs() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let hidden = 8usize;
    let inter = 8usize;
    let num_experts = 4usize;
    let num_tokens = 2usize;
    let top_k = 2usize;
    let block_size = 1usize;

    let x: Vec<f32> = (0..(num_tokens * hidden)).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let gate_w: Vec<f32> = (0..(num_experts * inter * hidden)).map(|i| ((i % 7) as f32) * 0.05).collect();
    let up_w: Vec<f32> = (0..(num_experts * inter * hidden)).map(|i| ((i % 5) as f32) * 0.05).collect();
    let down_w: Vec<f32> = (0..(num_experts * hidden * inter)).map(|i| ((i % 3) as f32) * 0.05).collect();

    let token_ids: Vec<u32> = vec![0, 0, 1, 1];
    let expert_ids: Vec<u32> = vec![0, 1, 2, 3];
    let weights: Vec<f32> = vec![0.6, 0.4, 0.7, 0.3];

    let (_x_s, x_ptr) = f32_dev(&dev, &x);
    let (_gw_s, gw_ptr) = f32_dev(&dev, &gate_w);
    let (_uw_s, uw_ptr) = f32_dev(&dev, &up_w);
    let (_dw_s, dw_ptr) = f32_dev(&dev, &down_w);

    let ti_bytes: Vec<u8> = token_ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ti_storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &ti_bytes,
            &Shape::new(vec![token_ids.len()]),
            DType { arith: ArithType::U32, storage: DTypeStorage::Native },
        )
        .expect("token_ids upload");
    let ti_dev_ptr = ti_storage.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().device_ptr_u64().unwrap();

    let ei_bytes: Vec<u8> = expert_ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ei_storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &ei_bytes,
            &Shape::new(vec![expert_ids.len()]),
            DType { arith: ArithType::U32, storage: DTypeStorage::Native },
        )
        .expect("expert_ids upload");
    let ei_dev_ptr = ei_storage.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().device_ptr_u64().unwrap();

    let (_w_s, w_ptr) = f32_dev(&dev, &weights);

    let (o1_s, o1_ptr) = f32_dev(&dev, &vec![0.0f32; num_tokens * hidden]);
    let (o2_s, o2_ptr) = f32_dev(&dev, &vec![0.0f32; num_tokens * hidden]);

    let mut graph_b = TaskGraphBuilder::new();
    let h_in = graph_b.push_arg(x_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_IN);
    let h_out1 = graph_b.push_arg(o1_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_OUT);
    let h_out2 = graph_b.push_arg(o2_ptr, (num_tokens * hidden * 4) as u64, arg_role::ACT_OUT);
    let h_gw = graph_b.push_arg(gw_ptr, (gate_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_uw = graph_b.push_arg(uw_ptr, (up_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_dw = graph_b.push_arg(dw_ptr, (down_w.len() * 4) as u64, arg_role::WEIGHT);
    let h_ti = graph_b.push_arg(ti_dev_ptr, (token_ids.len() * 4) as u64, arg_role::SCRATCH);
    let h_ei = graph_b.push_arg(ei_dev_ptr, (expert_ids.len() * 4) as u64, arg_role::SCRATCH);
    let h_w = graph_b.push_arg(w_ptr, (weights.len() * 4) as u64, arg_role::SCRATCH);

    let params = [
        0u32, hidden as u32, inter as u32, num_tokens as u32, block_size as u32,
        num_experts as u32, top_k as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    graph_b.push(op::MOE_DISPATCH, params, &[h_in, h_out1, h_gw, h_uw, h_dw, h_ti, h_ei, h_w], &[], 0, 0);
    graph_b.push(op::MOE_DISPATCH, params, &[h_in, h_out2, h_gw, h_uw, h_dw, h_ti, h_ei, h_w], &[], 0, 0);
    let graph = graph_b.finish();

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds moes");

    ctx.enqueue_dynamic_moe(&dev, &ctx.plan().moes()[0]).expect("enqueue dynamic moe 1");
    ctx.enqueue_dynamic_moe(&dev, &ctx.plan().moes()[1]).expect("enqueue dynamic moe 2");
    dev.synchronize();

    let r1: Vec<f32> = o1_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    let r2: Vec<f32> = o2_s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap().to_cpu_vec_f32().unwrap();
    assert_eq!(r1, r2, "two runs of on-device dynamic moe dispatch must be byte-identical");

    // NON-VACUITY, as above: the dynamic (Event-Tensor) enqueue must actually
    // produce signal, not leave the output at its initial value.
    assert!(
        r1.iter().any(|&x| x != 0.0),
        "dynamic moe dispatch produced all zeros — the on-device enqueue never ran"
    );
}

/// The readiness predicate advertises `RING_OPCODES`, and `WAVE_OPCODES` maps each
/// of them to a wave opcode. Both halves must be true OF THE KERNEL, and both have
/// been violated: `Q8_0_DOT4` was advertised with no wave arm at all, and the
/// number it was given (8) is the wave's `OP_PEER_REDUCE` — so had it ever been
/// bound, the ring would have run an all-reduce instead of a quantized GEMM.
///
/// This parses the kernel's own source, so the one hand-maintained list cannot
/// drift from the kernel it claims to describe.
#[test]
fn every_advertised_ring_opcode_maps_to_an_implemented_wave_arm() {
    use grim_backend_rocm::kernels::task_ring::{wave_opcode, WAVE_OPCODES};
    use grim_backend_rocm::task_graph_exec::RING_OPCODES;

    let src = grim_backend_rocm::kernels::scythe_persistent::KERNEL_SOURCE;

    // wave opcode number -> macro name, from the kernel's own `#define OP_X <n>u`.
    let mut by_number: std::collections::BTreeMap<u32, String> = Default::default();
    for line in src.lines() {
        let Some(rest) = line.trim().strip_prefix("#define OP_") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (Some(name), Some(value)) = (it.next(), it.next()) else {
            continue;
        };
        if let Some(v) = value.strip_suffix('u') {
            if let Ok(n) = v.parse::<u32>() {
                by_number.insert(n, format!("OP_{name}"));
            }
        }
    }
    assert!(
        by_number.len() >= 8,
        "parsed too few opcode defines — the parse is wrong, not the kernel: {by_number:?}"
    );

    for &(ir_op, wave) in WAVE_OPCODES {
        let macro_name = by_number
            .get(&wave)
            .unwrap_or_else(|| panic!("WAVE_OPCODES maps IR op {ir_op} to wave {wave}, which has no #define"));
        assert!(
            src.contains(&format!("desc->opcode == {macro_name}")),
            "WAVE_OPCODES maps IR op {ir_op} to wave {wave} ({macro_name}), but the kernel \
             never dispatches on it — the ring would silently execute nothing"
        );
    }

    // Everything the readiness predicate advertises must be in that mapping, or
    // `ring_ready` reports a model as fully covered while nodes fall back.
    for &op in RING_OPCODES {
        assert!(
            wave_opcode(op).is_some(),
            "RING_OPCODES advertises {op} but WAVE_OPCODES has no entry for it"
        );
    }

    // Regression net for the specific defect the scan above CANNOT see: the old
    // `Q8_0_DOT4 -> 8` entry pointed at a number the kernel DOES dispatch — as
    // `OP_PEER_REDUCE`. So "has an arm" was satisfied while the semantics were
    // wrong (an all-reduce for a quantized GEMM). Re-admitting it requires a real
    // dot4 arm, which is why this is pinned by name.
    assert!(
        wave_opcode(grim_backend_rocm::task_graph::op::Q8_0_DOT4).is_none(),
        "Q8_0_DOT4 must stay unmapped until a dot4 arm exists — its old number (8) is \
         OP_PEER_REDUCE, so mapping it dispatches the wrong kernel"
    );
}

/// M6 Step 1: `op::PEER_COPY` must BIND (it names an `ACT_IN` source and an
/// `arg_role::PEER` destination) and EXECUTE. The wave's `OP_PEER_BROADCAST` arm
/// writes `desc->input_ptr` into `desc->peer_ptr`, so pointing `peer_ptr` at a
/// local buffer makes this a plain copy — testable on ONE device, and it is
/// exactly the operation the cross-device handoff performs over peer access.
///
/// This is the gate that keeps `PEER_COPY` honest the way `Q8_0_DOT4` was not: it
/// is in `RING_OPCODES` only because it binds AND runs.
#[test]
fn peer_copy_binds_and_executes() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let src: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.25).collect();
    let (_s_s, s_ptr) = f32_dev(&dev, &src);
    let (d_s, d_ptr) = f32_dev(&dev, &[0.0f32; 16]);

    let mut gb = TaskGraphBuilder::new();
    let h_s = gb.push_arg(s_ptr, 64, arg_role::ACT_IN);
    let h_d = gb.push_arg(d_ptr, 64, arg_role::PEER);
    // params[0] = element count.
    gb.push(
        op::PEER_COPY,
        [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_s, h_d],
        &[],
        0,
        0,
    );
    let graph = gb.finish();
    assert!(graph.caps.needs_peer, "a PEER_COPY node must set needs_peer");

    let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
        .expect("ring context")
        .expect("ring binds the peer copy");
    assert_eq!(ctx.plan().peers().len(), 1, "the PEER_COPY node must bind");

    ctx.enqueue_peer_copy(&dev, &ctx.plan().peers()[0])
        .expect("enqueue peer copy");
    dev.synchronize();

    let got: Vec<f32> = d_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap()
        .to_cpu_vec_f32()
        .unwrap();
    assert_eq!(got, src, "the peer copy must reproduce the source exactly");
}

/// M6: the ring performs a REAL cross-device copy. The wave's `OP_PEER_BROADCAST`
/// arm writes `desc->input_ptr` (this device) into `desc->peer_ptr` (the peer
/// device's buffer), so with peer access granted the segment-boundary handoff
/// runs inside the ring — which is the point of M6, and is what a plain memcpy
/// node cannot do.
///
/// The single-device gate above proves the BINDING and the arm; this proves the
/// arm works across devices: the producer's wave must land the bytes in the
/// CONSUMER's buffer, not in some local scratch.
#[test]
fn ring_peer_copy_crosses_devices() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let producer = grim_backend_rocm::RocmDevice::try_new(0).expect("dev 0");
    let consumer = grim_backend_rocm::RocmDevice::try_new(1).expect("dev 1");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // The grant is directional: the wave runs on the PRODUCER and writes into
    // the CONSUMER's memory, so producer -> consumer is the direction that
    // matters.
    grim_backend_rocm::peer_access::enable_peer_access(0, 1).expect("peer 0->1");

    let src: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let (_s_s, s_ptr) = f32_dev(&producer, &src);
    let (d_s, d_ptr) = f32_dev(&consumer, &[0.0f32; 32]);

    let mut gb = TaskGraphBuilder::new();
    let h_s = gb.push_arg(s_ptr, 128, arg_role::ACT_IN);
    let h_d = gb.push_arg(d_ptr, 128, arg_role::PEER);
    gb.push(
        op::PEER_COPY,
        [32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_s, h_d],
        &[],
        0,
        0,
    );
    let graph = gb.finish();

    let ctx =
        grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&producer, &graph)
            .expect("ring context")
            .expect("ring binds the peer copy");
    ctx.enqueue_peer_copy(&producer, &ctx.plan().peers()[0])
        .expect("enqueue peer copy");

    // The write is issued on the producer's stream into the consumer's buffer.
    producer.synchronize();
    consumer.synchronize();

    let got: Vec<f32> = d_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap()
        .to_cpu_vec_f32()
        .unwrap();
    assert_eq!(
        got, src,
        "the ring's peer copy must land the bytes in the CONSUMER's buffer"
    );
}

/// Gate 1: the plan builder's binding rules.
#[test]
fn the_plan_binds_only_what_a_ring_can_run() {
    // Three GEMM nodes: one fully specified native-f32 GEMM, one quantized, one
    // missing its weight operand. Only the first is ring-executable.
    let a_ptr = 0x1000u64;
    let w_ptr = 0x2000u64;
    let o_ptr = 0x3000u64;
    let w_ptr2 = 0x4000u64;
    let o_ptr2 = 0x5000u64;

    let mut b = TaskGraphBuilder::new();
    // Node 0: executable — native f32, all three operands, real geometry.
    let h_in = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_w = b.push_arg(w_ptr, 64, arg_role::WEIGHT);
    let h_o = b.push_arg(o_ptr, 64, arg_role::ACT_OUT);
    let n0 = b.push(
        op::GEMM,
        [0, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_in, h_w, h_o],
        &[],
        0,
        0,
    );
    // Node 1: quantized (params[0] != 0) — the f32 wave arm cannot reproduce it.
    let h_in1 = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_w1 = b.push_arg(w_ptr, 64, arg_role::WEIGHT);
    let h_o1 = b.push_arg(o_ptr, 64, arg_role::ACT_OUT);
    let _n1 = b.push(
        op::GEMM,
        [5, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_in1, h_w1, h_o1],
        &[n0],
        0,
        0,
    );
    // Node 2: no weight operand at all.
    let h_in2 = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_o2 = b.push_arg(o_ptr2, 64, arg_role::ACT_OUT);
    let _n2 = b.push(
        op::GEMM,
        [0, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_in2, h_o2],
        &[n0],
        0,
        0,
    );
    // Node 3: zero geometry.
    let h_in3 = b.push_arg(a_ptr, 64, arg_role::ACT_IN);
    let h_w3 = b.push_arg(w_ptr2, 64, arg_role::WEIGHT);
    let h_o3 = b.push_arg(o_ptr2, 64, arg_role::ACT_OUT);
    b.push(
        op::GEMM,
        [0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_in3, h_w3, h_o3],
        &[n0],
        0,
        0,
    );

    let ir = b.finish();
    assert_eq!(ir.nodes.len(), 4, "four GEMM nodes emitted");
    let plan = TaskRingPlan::from_graph(&ir);
    assert_eq!(
        plan.len(),
        1,
        "only the fully-specified native-f32 GEMM is ring-executable: {:?}",
        plan
    );
    let g = plan.gemms[0];
    assert_eq!((g.m, g.n, g.k), (1, 4, 3), "geometry from params[1..4]");
    assert_eq!(
        (g.input, g.weight, g.output),
        (a_ptr, w_ptr, o_ptr),
        "operands bound by ROLE, not by position"
    );
    assert_eq!(g.quant, 0, "the wave's f32 arm is what this node ran on");

    // A non-GEMM node is never considered, whatever it carries.
    let mut b2 = TaskGraphBuilder::new();
    let h = b2.push_arg(w_ptr, 64, arg_role::WEIGHT);
    b2.push(
        op::ATTENTION,
        [0, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h],
        &[],
        0,
        0,
    );
    assert!(
        TaskRingPlan::from_graph(&b2.finish()).is_empty(),
        "a non-GEMM node must never enter the plan"
    );

    // The staging cap is a documented bound, not an accident: the largest in-tree
    // emitter (40 layers x FFN + head) must fit.
    assert!(
        STAGED_CAPACITY >= 64,
        "the largest in-tree emitter (40 layers x FFN + head) must fit"
    );
}

/// `&RocmStorage` as the `&dyn BackendStorage` the ring API takes. The
/// `scythe_route` path stores `Box<dyn BackendStorage>` directly; a raw
/// `RocmStorage` needs the unsizing coercion spelled out.
fn bs(s: &grim_backend_rocm::RocmStorage) -> &dyn BackendStorage {
    s as &dyn BackendStorage
}

/// A device-resident f32 tensor plus its device address.
///
/// The storage is returned alongside the pointer rather than converted to a
/// `RocmStorage` because `RocmStorage` is not `Clone`; the caller keeps the box
/// alive for as long as the wave may read it.
fn f32_dev(dev: &grim_backend_rocm::RocmDevice, data: &[f32]) -> (Box<dyn BackendStorage>, u64) {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &bytes,
            &Shape::new(vec![data.len()]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
        )
        .expect("f32 device tensor");
    let ptr = storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("f32 device tensor is RocmStorage")
        .device_ptr_u64()
        .expect("f32 device tensor has a device ptr");
    (storage, ptr)
}

/// A zeroed device u32 scalar plus its address.
fn u32_dev(dev: &grim_backend_rocm::RocmDevice, v: u32) -> (Box<dyn BackendStorage>, u64) {
    let storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &v.to_ne_bytes(),
            &Shape::new(vec![1]),
            DType {
                arith: ArithType::U32,
                storage: DTypeStorage::Native,
            },
        )
        .expect("u32 device scalar");
    let ptr = storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("u32 device scalar is RocmStorage")
        .device_ptr_u64()
        .expect("u32 device scalar has a device ptr");
    (storage, ptr)
}

/// Gate 2: one wave launch runs a step's GEMM set, and it computes correctly.
///
/// Decode-shaped GEMMs (m = 1) chained so the second consumes the first's
/// output — the dependency the plan's ordering exists to honour.
#[test]
fn one_wave_launch_runs_the_steps_gemms() {
    if std::env::var("GRIM_GPU_TEST").map(|v| v != "1").unwrap_or(true) {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the task-ring execution gate");
        return;
    }
    let _guard = DeviceGuard::set(0);
    let dev = grim_backend_rocm::RocmDevice::shared(0);

    // C1 = A1 @ B1^T (m=1, n=4, k=3), then C2 = C1 @ B2^T (m=1, n=2, k=4).
    let a1 = [1.0f32, 2.0, 3.0];
    let b1 = [
        1.0f32, 0.0, 1.0, // [N=4, K=3] row-major
        0.0, 1.0, 1.0,
        1.0, 1.0, 0.0,
        0.5, 0.5, 0.5,
    ];
    let b2 = [
        1.0f32, 0.0, 0.0, 1.0, // [N=2, K=4] row-major
        0.0, 1.0, 1.0, 0.0,
    ];

    // Host reference for the chain.
    let c1_ref: Vec<f32> = (0..4)
        .map(|n| (0..3).map(|k| a1[k] * b1[n * 3 + k]).sum())
        .collect();
    let c2_ref: Vec<f32> = (0..2)
        .map(|n| (0..4).map(|k| c1_ref[k] * b2[n * 4 + k]).sum())
        .collect();

    // The boxes must outlive the wave that reads them; the pointers are what the
    // plan binds.
    let (_a1_s, a1_p) = f32_dev(&dev, &a1);
    let (_b1_s, b1_p) = f32_dev(&dev, &b1);
    let (c1_s, c1_p) = f32_dev(&dev, &[0.0f32; 4]);
    let (_b2_s, b2_p) = f32_dev(&dev, &b2);
    let (c2_s, c2_p) = f32_dev(&dev, &[0.0f32; 2]);

    // The plan the interpreter would build: two chained native GEMMs.
    let plan = TaskRingPlan {
        norms: Vec::new(),
        adds: Vec::new(),
        attns: Vec::new(),
        moes: Vec::new(),
        peers: Vec::new(),
        recurrences: Vec::new(),
        gemms: vec![
            TaskRingGemm {
                m: 1,
                n: 4,
                k: 3,
                input: a1_p,
                weight: b1_p,
                output: c1_p,
                quant: 0,
                // Standalone plan entries have no IR to point at.
                ir_node: u32::MAX,
            },
            TaskRingGemm {
                m: 1,
                n: 2,
                k: 4,
                input: c1_p,
                weight: b2_p,
                output: c2_p,
                quant: 0,
                ir_node: u32::MAX,
            },
        ],
    };

    // Stage once (the geometry of a decode step is constant between steps).
    let (staged, count) = plan.stage_on_device(&dev).expect("stage plan");
    assert_eq!(count, 2, "both descriptors staged");

    // A ring: 8 slots, tail/head/stop scalars zeroed.
    let cap: u32 = 8;
    let slots = dev
        .alloc_scythe_ring_bytes((cap as usize) * 64)
        .expect("ring slots");
    let (tail_s, _) = u32_dev(&dev, 0);
    let (head_s, _) = u32_dev(&dev, 0);
    let (stop_s, _) = u32_dev(&dev, 0);

    let before = wave_launch_count();
    launch_task_ring(
        &dev,
        bs(&slots),
        cap,
        tail_s.as_ref(),
        head_s.as_ref(),
        stop_s.as_ref(),
        Some(&*staged),
        count,
    )
    .expect("ring launch")
    .synchronize()
    .expect("wave completes");
    let after = wave_launch_count();

    // THE ASSERTION THE PHASE EXISTS FOR: two GEMMs, one launch.
    assert_eq!(
        after - before,
        1,
        "the step's whole GEMM set must cost ONE wave launch (was {})",
        after - before
    );

    let got_c1: Vec<f32> = c1_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("c1 is RocmStorage")
        .to_cpu_vec_f32()
        .expect("c1 readback");
    let got_c2: Vec<f32> = c2_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("c2 is RocmStorage")
        .to_cpu_vec_f32()
        .expect("c2 readback");
    for (i, (g, r)) in got_c1.iter().zip(c1_ref.iter()).enumerate() {
        assert!(
            (g - r).abs() <= 1e-5 * r.abs().max(1.0),
            "c1[{i}]: ring {g} vs reference {r}"
        );
    }
    for (i, (g, r)) in got_c2.iter().zip(c2_ref.iter()).enumerate() {
        assert!(
            (g - r).abs() <= 1e-4 * r.abs().max(1.0),
            "c2[{i}]: ring {g} vs reference {r} (chained through c1)"
        );
    }

    // A staged array with count 0 is a caller bug: refused at the boundary,
    // not at step time.
    let (staged0, _) = f32_dev(&dev, &[0.0f32; 8]);
    let r = launch_task_ring(
        &dev,
        bs(&slots),
        cap,
        tail_s.as_ref(),
        head_s.as_ref(),
        stop_s.as_ref(),
        Some(&*staged0),
        0,
    );
    assert!(r.is_err(), "staged_count 0 with a staged array must be refused");
}

/// M5: the ring's OP_RECURRENT arm must run the qwen35 KDA/recurrent step
/// IDENTICALLY to the three kernels it replaces (short-conv causal step, batched
/// gated delta rule, gated head norm). The standalone kernels are the oracle —
/// the arm is the thing under test — so a no-op arm fails this on step 0, and an
/// arm that drops a pass fails on the state it would leave behind.
///
/// Two steps with the same inputs: the state carries, so a stale-state arm passes
/// step 0 and diverges on step 1 (the same trap `kda_batched_parity` exists for).
#[test]
fn ring_recurrent_matches_the_standalone_kernels() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = grim_backend_rocm::RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    const NV: usize = 6;
    const NK: usize = 2;
    const D: usize = 16;
    const KS: usize = 4;
    const CHANNELS: usize = 2 * NK * D + NV * D;
    const EPS: f32 = 1e-6;

    let mut seed: u64 = 0x1234_5678;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let conv_in: Vec<f32> = (0..CHANNELS).map(|_| rnd() * 0.5).collect();
    let conv_w: Vec<f32> = (0..CHANNELS * KS).map(|_| rnd() * 0.2).collect();
    let conv_state: Vec<f32> = (0..CHANNELS * (KS - 1)).map(|_| rnd() * 0.1).collect();
    let alpha: Vec<f32> = (0..NV).map(|_| rnd() * 0.3).collect();
    let beta: Vec<f32> = (0..NV).map(|_| rnd() * 0.3).collect();
    let dt_bias: Vec<f32> = (0..NV).map(|_| rnd() * 0.3).collect();
    let ssm_a: Vec<f32> = (0..NV).map(|_| rnd() * 0.5).collect();
    let ssm_norm: Vec<f32> = (0..D).map(|_| 1.0 + rnd() * 0.01).collect();
    let gate: Vec<f32> = (0..NV * D).map(|_| rnd() * 0.5).collect();
    let state0: Vec<f32> = (0..NV * D * D).map(|_| rnd() * 0.3).collect();

    // ── Ring arm: one descriptor, two steps, state carried in place.
    let (_ci_s, ci_ptr) = f32_dev(&dev, &conv_in);
    let (_cw_s, cw_ptr) = f32_dev(&dev, &conv_w);
    let (_cs_ring_s, cs_ring_ptr) = f32_dev(&dev, &conv_state);
    let (_co_ring_s, co_ring_ptr) = f32_dev(&dev, &[0.0f32; CHANNELS]);
    let (a_s, a_ptr) = f32_dev(&dev, &alpha);
    let (b_s, b_ptr) = f32_dev(&dev, &beta);
    let (d_s, d_ptr) = f32_dev(&dev, &dt_bias);
    let (sa_s, sa_ptr) = f32_dev(&dev, &ssm_a);
    let (sn_s, sn_ptr) = f32_dev(&dev, &ssm_norm);
    let (g_s, g_ptr) = f32_dev(&dev, &gate);
    let (st_ring_s, st_ring_ptr) = f32_dev(&dev, &state0);
    let (_acc_ring_s, acc_ring_ptr) = f32_dev(&dev, &[0.0f32; NV * D]);
    let (br_ring_s, br_ring_ptr) = f32_dev(&dev, &[0.0f32; NV * D]);

    let desc = grim_backend_rocm::scythe_task_descriptors::KdaTaskDescriptor {
        conv_in: ci_ptr,
        conv_w: cw_ptr,
        conv_bias: 0,
        conv_state: cs_ring_ptr,
        conv_out: co_ring_ptr,
        alpha: a_ptr,
        beta: b_ptr,
        dt_bias: d_ptr,
        ssm_a: sa_ptr,
        ssm_norm: sn_ptr,
        gate: g_ptr,
        state: st_ring_ptr,
        acc: acc_ring_ptr,
        branch: br_ring_ptr,
        num_v: NV as u32,
        num_k: NK as u32,
        head_dim: D as u32,
        batch: 1,
        channels: CHANNELS as u32,
        kernel_size: KS as u32,
        eps: EPS,
        has_z: 1,
    };
    let kda_dev = desc
        .upload(&dev)
        .expect("upload KdaTaskDescriptor");

    // ── Standalone oracle: separate copies, same inputs.
    let (cs_ref_s, _cs_ref_ptr) = f32_dev(&dev, &conv_state);
    let (co_ref_s, _co_ref_ptr) = f32_dev(&dev, &[0.0f32; CHANNELS]);
    let (st_ref_s, _st_ref_ptr) = f32_dev(&dev, &state0);
    let (acc_ref_s, _acc_ref_ptr) = f32_dev(&dev, &[0.0f32; NV * D]);
    let (br_ref_s, _br_ref_ptr) = f32_dev(&dev, &[0.0f32; NV * D]);

    let mut gb = TaskGraphBuilder::new();
    let h_desc = gb.push_arg(kda_dev, 144, arg_role::WEIGHT);
    gb.push(op::RECURRENT, [0; 16], &[h_desc], &[], 0, 0);
    let graph = gb.finish();
    let ctx =
        grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
            .expect("ring context")
            .expect("ring binds the recurrent node");
    assert_eq!(ctx.plan().recurrences().len(), 1, "the RECURRENT node must bind");

    let mut ring_branches: Vec<Vec<f32>> = Vec::new();
    let mut ring_states: Vec<Vec<f32>> = Vec::new();
    let rd = |s: &Box<dyn BackendStorage>| -> Vec<f32> {
        s.as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap()
            .to_cpu_vec_f32()
            .unwrap()
    };
    for _step in 0..2 {
        ctx.enqueue_recurrent(&dev, &ctx.plan().recurrences()[0])
            .expect("enqueue recurrent");
        dev.synchronize();
        ring_branches.push(rd(&br_ring_s));
        ring_states.push(rd(&st_ring_s));
    }

    // The standalone oracle runs the SAME two steps on its own buffer copies.
    let mut ref_branches: Vec<Vec<f32>> = Vec::new();
    let mut ref_states: Vec<Vec<f32>> = Vec::new();
    fn rocm_of(s: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        s.as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage")
    }
    for _step in 0..2 {
        dev.short_conv1d_causal_step_into(
            rocm_of(&_ci_s),
            rocm_of(&_cw_s),
            None,
            rocm_of(&cs_ref_s),
            rocm_of(&co_ref_s),
        )
        .expect("standalone short conv");
        dev.kda_gated_delta_rule_batched_into(
            rocm_of(&co_ref_s),
            rocm_of(&a_s),
            rocm_of(&b_s),
            rocm_of(&d_s),
            rocm_of(&sa_s),
            rocm_of(&sn_s),
            Some(rocm_of(&g_s)),
            rocm_of(&st_ref_s),
            rocm_of(&acc_ref_s),
            rocm_of(&br_ref_s),
            NV,
            NK,
            D,
            EPS,
        )
        .expect("standalone KDA");
        dev.synchronize();
        ref_branches.push(rd(&br_ref_s));
        ref_states.push(rd(&st_ref_s));
    }

    for step in 0..2 {
        assert_eq!(
            ring_branches[step],
            ref_branches[step],
            "step {step}: the ring's recurrent branch differs from the standalone kernels"
        );
        assert_eq!(
            ring_states[step],
            ref_states[step],
            "step {step}: the ring's recurrent STATE differs — a stale-state arm passes step 0"
        );
    }
}
