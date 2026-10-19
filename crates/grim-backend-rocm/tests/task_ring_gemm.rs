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
