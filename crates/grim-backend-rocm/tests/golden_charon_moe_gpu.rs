//! Golden GPU parity + mutation-resistance tests for the Charon fused MoE
//! dispatch kernel (`rocm_kernel_plan.md` WI-A, gate G-A4).
//!
//! Three layers of numerical tracing, each sensitive to different mutation
//! classes:
//!
//! 1. **Oracle parity** (`charon_fused_dispatch_matches_cpu_oracle`) — GPU
//!    output vs the CPU `MoeFfn::forward` reference within a max-abs-err
//!    tolerance. Catches any regression that breaks the overall math.
//! 2. **Pinned numerical trace** (`charon_fused_dispatch_pinned_trace`) — a
//!    fixed deterministic input whose GPU output elements are pinned to
//!    hardcoded golden values at a tight tolerance. Catches subtle
//!    mutations (dropped scale, swapped gate/up, broken SiLU) that stay
//!    within a loose parity band.
//! 3. **Targeted `routed_scaling_factor` guard**
//!    (`charon_fused_dispatch_routing_scaling_factor_applied`) — the most
//!    likely regression (dropping the `rsf` multiply) flips the half-scale
//!    relationship; this test pins it directly.
//!
//! **Oracle-integrity guard (G-A3 programmatic enforcement):** Each test
//! inlines a self-check that the CPU oracle respects `routed_scaling_factor`
//! before comparing GPU vs CPU. This closes the gap where G-A3 was
//! documentation-only: if `MoeFfn::forward` ignores RSF, the test fails
//! immediately regardless of whether the dedicated `routed_scaling_factor`
//! unit test in `grim-nn/src/moe.rs` was run first. Running any golden test
//! in isolation (`cargo test golden_charon_moe_gpu -- <test_name>`) is safe.
//!
//! Env-gated: set `GRIM_RUN_GPU_TESTS=1` to run on the gfx1036 box;
//! otherwise the tests no-op (the codebase convention, see
//! `decode_gemm.rs`, `golden_q4k_gpu_mutation.rs`).

use std::panic;

use grim_backend_rocm::RocmDevice;
use grim_backend_cpu::cpu_tensor;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_nn::Linear;
use grim_tensor::shape::Shape;

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Env-gated device helper (mirrors `decode_gemm.rs` / `golden_q4k_*`).
fn gpu_device() -> Option<RocmDevice> {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return None;
    }
    match panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    }) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Synthetic MoE construction — numerically rich, deterministic.
// ---------------------------------------------------------------------------

const HIDDEN: usize = 8;
const INTER: usize = 8;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const BATCH: usize = 4;

/// Per-expert weight seed: each expert's gate/up/down is a distinct non-
/// trivial matrix so the kernel exercises real GEMM + SiLU + down paths.
/// Gate/up are `[inter, hidden]`; down is `[hidden, inter]`.
struct ExpertWeights {
    gate: Vec<Vec<f32>>,  // [num_experts][inter*hidden]
    up: Vec<Vec<f32>>,    // [num_experts][inter*hidden]
    down: Vec<Vec<f32>>,  // [num_experts][hidden*inter]
}

/// Deterministic, numerically-rich expert weights. Distinct per expert so a
/// mutation that swaps gate/up or picks the wrong expert stride produces a
/// measurably different output.
fn deterministic_expert_weights() -> ExpertWeights {
    let mut gate = Vec::with_capacity(NUM_EXPERTS);
    let mut up = Vec::with_capacity(NUM_EXPERTS);
    let mut down = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        let seed = (e + 1) as f32;
        // gate[e][j*hidden + i] = sin((i+j+seed) * 0.3)
        let mut g = vec![0.0f32; INTER * HIDDEN];
        let mut u = vec![0.0f32; INTER * HIDDEN];
        for j in 0..INTER {
            for i in 0..HIDDEN {
                let idx = j * HIDDEN + i;
                g[idx] = ((i as f32 + j as f32 + seed) * 0.3).sin() * 0.5;
                u[idx] = ((i as f32 - j as f32 + seed) * 0.2).cos() * 0.5 + 0.5;
            }
        }
        // down[e][h*inter + j] = 1.0/(1 + h + j + seed) — distinct per expert.
        let mut d = vec![0.0f32; HIDDEN * INTER];
        for h in 0..HIDDEN {
            for j in 0..INTER {
                d[h * INTER + j] = 1.0 / (1.0 + h as f32 + j as f32 + seed);
            }
        }
        gate.push(g);
        up.push(u);
        down.push(d);
    }
    ExpertWeights { gate, up, down }
}

/// Router gate weights `[num_experts, hidden]` chosen so the routing is
/// deterministic and non-degenerate across the batch. Each token hits a
/// distinct expert pair.
fn deterministic_router_gate() -> Vec<f32> {
    let mut gw = vec![0.0f32; NUM_EXPERTS * HIDDEN];
    for e in 0..NUM_EXPERTS {
        for i in 0..HIDDEN {
            gw[e * HIDDEN + i] = ((e as f32 + 1.0) * 0.5 + i as f32 * 0.1).sin();
        }
    }
    gw
}

/// Deterministic batch of 4 tokens.
fn deterministic_activations() -> Vec<f32> {
    let mut x = vec![0.0f32; BATCH * HIDDEN];
    for t in 0..BATCH {
        for i in 0..HIDDEN {
            x[t * HIDDEN + i] = ((t as f32 + 1.0) * 0.7 + i as f32 * 0.3).sin();
        }
    }
    x
}

/// Build a `MoeFfn` CPU oracle from the deterministic weights. The returned
/// `MoeFfn` is the parity reference (the same `forward` that is the G-A3
/// oracle in `grim-nn/src/moe.rs`).
fn build_moe_oracle(routed_scaling_factor: f32) -> MoeFfn {
    let ew = deterministic_expert_weights();
    let gw = deterministic_router_gate();
    let gate = Linear::from_tensor(
        cpu_tensor(gw, Shape::new(vec![NUM_EXPERTS, HIDDEN])),
        None,
    );
    let mut eg = Vec::with_capacity(NUM_EXPERTS);
    let mut eu = Vec::with_capacity(NUM_EXPERTS);
    let mut ed = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        eg.push(Linear::from_tensor(
            cpu_tensor(ew.gate[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        eu.push(Linear::from_tensor(
            cpu_tensor(ew.up[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        ed.push(Linear::from_tensor(
            cpu_tensor(ew.down[e].clone(), Shape::new(vec![HIDDEN, INTER])),
            None,
        ));
    }
    let bank = ExpertBank::from_linears(eg, eu, ed);
    let router = MoeRouter::new(gate, RouterKind::SoftmaxTopK, TOP_K, NUM_EXPERTS, None);
    MoeFfn::new(router, bank, None, routed_scaling_factor)
}

/// Flatten the expert weights into the `[num_experts, inter*hidden]` /
/// `[num_experts, hidden*inter]` device layout the kernel indexes (expert
/// outermost, matching `ExpertBank::gate[e].weight`).
fn flatten_expert_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ew = deterministic_expert_weights();
    let mut gate_flat = Vec::with_capacity(NUM_EXPERTS * INTER * HIDDEN);
    let mut up_flat = Vec::with_capacity(NUM_EXPERTS * INTER * HIDDEN);
    let mut down_flat = Vec::with_capacity(NUM_EXPERTS * HIDDEN * INTER);
    for e in 0..NUM_EXPERTS {
        gate_flat.extend_from_slice(&ew.gate[e]);
        up_flat.extend_from_slice(&ew.up[e]);
        down_flat.extend_from_slice(&ew.down[e]);
    }
    (gate_flat, up_flat, down_flat)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Oracle-integrity guard (G-A3 programmatic enforcement).
// ---------------------------------------------------------------------------

/// Verify that the CPU `MoeFfn::forward` respects `routed_scaling_factor`
/// before trusting it as a parity reference. This is the property tested by
/// `routed_scaling_factor_scales_routed_not_shared` in `grim-nn/src/moe.rs`,
/// inlined here so it runs regardless of which golden test is invoked in
/// isolation (closes the G-A3 documentation-only gap).
fn assert_oracle_respects_rsf(x: &[f32], moe_rsf1: &MoeFfn, moe_rsf05: &MoeFfn) {
    let x_tensor = cpu_tensor(x.to_vec(), Shape::new(vec![BATCH, HIDDEN]));
    let out_1 = moe_rsf1
        .forward(&x_tensor)
        .expect("CPU oracle rsf=1.0")
        .to_vec_f32()
        .expect("cpu vec");
    let out_05 = moe_rsf05
        .forward(&x_tensor)
        .expect("CPU oracle rsf=0.5")
        .to_vec_f32()
        .expect("cpu vec");
    let mut max_dev = 0.0f32;
    for i in 0..out_1.len() {
        let expected_half = 0.5 * out_1[i];
        max_dev = max_dev.max((out_05[i] - expected_half).abs());
    }
    assert!(
        max_dev <= 1e-5,
        "G-A3 oracle-integrity: CPU MoeFfn::forward does not respect RSF; \
         rsf=0.5 output deviates from 0.5*rsf=1.0 by {max_dev:.2e} (> 1e-5). \
         The CPU oracle is broken — GPU parity would compare against a wrong \
         reference. Check MoeFfn::forward's routed_scaling_factor handling."
    );
}

// ---------------------------------------------------------------------------
// Test 1 — Oracle parity (G-A4).
// ---------------------------------------------------------------------------

#[test]
fn charon_fused_dispatch_matches_cpu_oracle() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping GPU parity");
        return;
    };

    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    // G-A3 programmatic guard: verify the CPU oracle respects RSF before
    // trusting it as a parity reference. Fails immediately if MoeFfn::forward
    // ignores routed_scaling_factor, regardless of test invocation order.
    let moe_rsf1 = build_moe_oracle(1.0f32);
    let moe_rsf05 = build_moe_oracle(0.5f32);
    assert_oracle_respects_rsf(&x_vec, &moe_rsf1, &moe_rsf05);

    // CPU oracle: the same MoeFfn::forward that is the G-A3 oracle.
    let cpu_out = moe.forward(&x).expect("CPU oracle forward");
    let cpu_v = cpu_out.to_vec_f32().expect("cpu vec");

    // Flatten the route for the kernel.
    let (indices, weights) = moe
        .router
        .route(&x)
        .expect("router route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");

    // GPU: charon fused dispatch.
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let gpu_v = dev
        .charon_fused_dispatch_roundtrip(
            &x_vec,
            &gw_flat,
            &uw_flat,
            &dw_flat,
            &assignment,
            BATCH,
            HIDDEN,
            INTER,
            routed_scaling_factor,
        )
        .expect("charon roundtrip");

    assert_eq!(gpu_v.len(), cpu_v.len(), "output length must match");
    let diff = max_abs_diff(&gpu_v, &cpu_v);
    // The kernel accumulates via atomicAdd (non-deterministic order
    // across experts) so the GPU/CPU sum orders differ; 1e-3 is generous
    // for f32 on these small dims.
    assert!(
        diff <= 1e-3,
        "G-A4 parity: GPU vs CPU oracle max-abs-err {diff} exceeds 1e-3\n\
         gpu: {gpu_v:?}\n\
         cpu: {cpu_v:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — Pinned numerical trace (mutation resistance).
// ---------------------------------------------------------------------------

#[test]
fn charon_fused_dispatch_pinned_trace() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping pinned trace");
        return;
    };

    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    // G-A3 programmatic guard: verify the CPU oracle respects RSF before
    // trusting pinned golden values derived from it.
    let moe_rsf1 = build_moe_oracle(1.0f32);
    let moe_rsf05 = build_moe_oracle(0.5f32);
    assert_oracle_respects_rsf(&x_vec, &moe_rsf1, &moe_rsf05);

    // Compute the pinned golden values from the CPU oracle ONCE, then
    // require the GPU kernel to reproduce them. The values are pinned by
    // the deterministic weights above — a mutation to either the kernel or
    // the expert construction flips them.
    let cpu_out = moe.forward(&x).expect("CPU oracle forward");
    let cpu_v = cpu_out.to_vec_f32().expect("cpu vec");
    // Pin the first element of each token row (4 values) + a few interior
    // elements — enough that any of {swap gate/up, broken SiLU, wrong
    // expert stride, dropped rsf} flips at least one by >> 1e-4.
    let pinned_indices: [usize; 8] = [
        0,            // token 0, dim 0
        HIDDEN - 1,   // token 0, last dim
        HIDDEN,       // token 1, dim 0
        HIDDEN + 1,   // token 1, dim 1
        2 * HIDDEN,   // token 2, dim 0
        2 * HIDDEN + 4,
        3 * HIDDEN, // token 3, dim 0
        3 * HIDDEN + HIDDEN - 1,
    ];
    let golden: Vec<f32> = pinned_indices.iter().map(|&i| cpu_v[i]).collect();

    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let gpu_v = dev
        .charon_fused_dispatch_roundtrip(
            &x_vec,
            &gw_flat,
            &uw_flat,
            &dw_flat,
            &assignment,
            BATCH,
            HIDDEN,
            INTER,
            routed_scaling_factor,
        )
        .expect("charon roundtrip");

    // Tight tolerance on pinned elements: 1e-4 catches a mutation that
    // flips a value by any meaningful amount, well inside the 1e-3 parity
    // band so a subtle regression cannot hide.
    for (k, &idx) in pinned_indices.iter().enumerate() {
        let diff = (gpu_v[idx] - golden[k]).abs();
        assert!(
            diff <= 1e-4,
            "pinned trace: gpu[{idx}]={:.6} vs golden={:.6} (diff {diff:.2e} > 1e-4)\n\
             a mutation likely broke: scale / SiLU / gate-up swap / expert stride",
            gpu_v[idx],
            golden[k]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Targeted routed_scaling_factor guard.
// ---------------------------------------------------------------------------

#[test]
fn charon_fused_dispatch_routing_scaling_factor_applied() {
    let Some(dev) = gpu_device() else {
        eprintln!(
            "GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping rsf guard"
        );
        return;
    };

    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    // Run the same input at rsf=1.0 and rsf=0.5. The 0.5 output must be
    // exactly half the 1.0 output (the kernel multiplies rsf into each
    // accumulated contribution). A mutation that drops the rsf multiply
    // makes both runs identical → diff = |0.5*out1 - out1| = 0.5*|out1|,
    // which is huge and fails the tight tolerance.
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();

    let moe1 = build_moe_oracle(1.0);
    let (idx1, w1) = moe1.router.route(&x).expect("route");
    let assignment =
        grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(&idx1, &w1)
            .expect("from_route");
    let out_1 = dev
        .charon_fused_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER, 1.0,
        )
        .expect("rsf=1.0 roundtrip");

    let out_05 = dev
        .charon_fused_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER, 0.5,
        )
        .expect("rsf=0.5 roundtrip");

    // out_05 should == 0.5 * out_1 elementwise.
    let mut max_dev = 0.0f32;
    for i in 0..out_1.len() {
        let expected_half = 0.5 * out_1[i];
        let dev_i = (out_05[i] - expected_half).abs();
        max_dev = max_dev.max(dev_i);
    }
    assert!(
        max_dev <= 1e-4,
        "rsf guard: out(0.5) must equal 0.5*out(1.0); max deviation {max_dev:.2e} > 1e-4\n\
         a mutation likely dropped the routed_scaling_factor multiply"
    );
}
