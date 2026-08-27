//! P2 device verifier — Charon MoE backward kernel produces correct
//! `d_gate_w`/`d_up_w`/`d_down_w`/`d_x` on a real ROCm device.
//!
//! Strategy:
//! 1. Build a small deterministic MoE (same geometry as the host-side
//!    `charon_backward_grad_check.rs`: HIDDEN=4, INTER=3, NUM_EXPERTS=2,
//!    TOP_K=1, BATCH=4, RSF=0.7).
//! 2. Route via `MoeRouter::route` → `RoutingAssignment`.
//! 3. Upload + launch `grim_moe_fused_grouped_backward` via
//!    `charon_grouped_backward_roundtrip`.
//! 4. Compare device grads against a host analytical backward reference
//!    (identical decomposition to `charon_backward_grad_check.rs`).
//! 5. Assert RMS relative error ≤ 0.01 for each of the four named grads.
//!
//! The host reference is NOT the kernel — it's an independent Rust
//! computation of the same math. Device-vs-host agreement at this
//! tolerance proves the HIP kernel executes the decomposition correctly.

use std::panic;

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::RocmDevice;
use grim_nn::Linear;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::shape::Shape;

// Same geometry as charon_backward_grad_check.rs.
const HIDDEN: usize = 4;
const INTER: usize = 3;
const NUM_EXPERTS: usize = 2;
const TOP_K: usize = 1;
const BATCH: usize = 4;
const RSF: f32 = 0.7;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

// ---------------------------------------------------------------------------
// Deterministic inputs (mirrors charon_backward_grad_check.rs flat_expert_weights
// but with BATCH=4 so d_x accumulation across tokens is exercised).
// ---------------------------------------------------------------------------

struct FlatWeights {
    gate: Vec<Vec<f32>>, // [num_experts][inter*hidden]
    up: Vec<Vec<f32>>,   // [num_experts][inter*hidden]
    down: Vec<Vec<f32>>, // [num_experts][hidden*inter]
    x: Vec<f32>,         // [batch*hidden]
}

fn flat_expert_weights() -> FlatWeights {
    let mut gate = Vec::with_capacity(NUM_EXPERTS);
    let mut up = Vec::with_capacity(NUM_EXPERTS);
    let mut down = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        let seed = (e as f32) * 0.5 + 0.3;
        let mut g = vec![0.0f32; INTER * HIDDEN];
        let mut u = vec![0.0f32; INTER * HIDDEN];
        for j in 0..INTER {
            for i in 0..HIDDEN {
                let idx = j * HIDDEN + i;
                g[idx] = ((i as f32 + j as f32 + seed) * 0.31).sin() * 0.4;
                u[idx] = ((i as f32 - j as f32 - seed) * 0.27).cos() * 0.4 + 0.1;
            }
        }
        let mut d = vec![0.0f32; HIDDEN * INTER];
        for h in 0..HIDDEN {
            for j in 0..INTER {
                d[h * INTER + j] = 0.3 / (1.0 + h as f32 + j as f32 + seed);
            }
        }
        gate.push(g);
        up.push(u);
        down.push(d);
    }
    // Batch of 4 tokens, all nonzero.
    let x: Vec<f32> = (0..BATCH * HIDDEN)
        .map(|k| ((k as f32 + 1.0) * 0.37).sin() * 0.5 + 0.2)
        .collect();
    FlatWeights { gate, up, down, x }
}

/// Router that deterministically picks expert 0 (well-separated logits so
/// TOP_K=1 is stable under expert-weight perturbations — same as
/// charon_backward_grad_check.rs).
fn well_separated_router_gate() -> Vec<f32> {
    let mut gw = vec![0.0f32; NUM_EXPERTS * HIDDEN];
    for i in 0..HIDDEN {
        gw[i] = 2.0 + 0.1 * i as f32;
        gw[HIDDEN + i] = -2.0 - 0.1 * i as f32;
    }
    gw
}

fn build_oracle(fw: &FlatWeights) -> MoeFfn {
    let gw = well_separated_router_gate();
    let router_gate =
        Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![NUM_EXPERTS, HIDDEN])), None);
    let router = MoeRouter::new(
        router_gate,
        RouterKind::SoftmaxTopK,
        TOP_K,
        NUM_EXPERTS,
        None,
    );
    let mut eg = Vec::with_capacity(NUM_EXPERTS);
    let mut eu = Vec::with_capacity(NUM_EXPERTS);
    let mut ed = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        eg.push(Linear::from_tensor(
            cpu_tensor(fw.gate[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        eu.push(Linear::from_tensor(
            cpu_tensor(fw.up[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        ed.push(Linear::from_tensor(
            cpu_tensor(fw.down[e].clone(), Shape::new(vec![HIDDEN, INTER])),
            None,
        ));
    }
    let bank = ExpertBank::from_linears(eg, eu, ed);
    MoeFfn::new(router, bank, None, RSF)
}

// ---------------------------------------------------------------------------
// Host analytical backward (same decomposition as charon_backward_grad_check.rs)
// ---------------------------------------------------------------------------

struct MoEGrads {
    d_x: Vec<f32>,
    d_gate_w: Vec<Vec<f32>>,
    d_up_w: Vec<Vec<f32>>,
    d_down_w: Vec<Vec<f32>>,
}

#[inline]
fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}
#[inline]
fn silu_grad(z: f32) -> f32 {
    let s = 1.0 / (1.0 + (-z).exp());
    s * (1.0 + z * (1.0 - s))
}

fn analytical_backward(fw: &FlatWeights) -> MoEGrads {
    let moe = build_oracle(fw);
    let x_tensor = cpu_tensor(fw.x.clone(), Shape::new(vec![BATCH, HIDDEN]));

    let (indices, weights) = moe.router.route(&x_tensor).expect("route");

    let mut d_x = vec![0.0f32; BATCH * HIDDEN];
    let mut d_gate_w: Vec<Vec<f32>> = (0..NUM_EXPERTS)
        .map(|_| vec![0.0; INTER * HIDDEN])
        .collect();
    let mut d_up_w: Vec<Vec<f32>> = (0..NUM_EXPERTS)
        .map(|_| vec![0.0; INTER * HIDDEN])
        .collect();
    let mut d_down_w: Vec<Vec<f32>> = (0..NUM_EXPERTS)
        .map(|_| vec![0.0; HIDDEN * INTER])
        .collect();

    // d_y = 1 for all (sum objective).
    let d_y = [1.0f32; HIDDEN];

    for t in 0..BATCH {
        let chosen = &indices[t];
        let w = &weights[t];

        for (rank, &e) in chosen.iter().enumerate() {
            let combine = w[rank];
            let s = RSF * combine;

            let x_row = &fw.x[t * HIDDEN..(t + 1) * HIDDEN];
            let gate_w = &fw.gate[e];
            let up_w = &fw.up[e];
            let down_w = &fw.down[e];

            // Forward recomputation.
            let mut h_gate = [0.0f32; INTER];
            let mut h_up = [0.0f32; INTER];
            for j in 0..INTER {
                for i in 0..HIDDEN {
                    h_gate[j] += gate_w[j * HIDDEN + i] * x_row[i];
                    h_up[j] += up_w[j * HIDDEN + i] * x_row[i];
                }
            }
            let act: Vec<f32> = (0..INTER).map(|j| silu(h_gate[j]) * h_up[j]).collect();

            // d_down_w + d_act.
            let mut d_act = [0.0f32; INTER];
            for h in 0..HIDDEN {
                let dyh_s = s * d_y[h];
                for j in 0..INTER {
                    d_down_w[e][h * INTER + j] += dyh_s * act[j];
                    d_act[j] += s * d_y[h] * down_w[h * INTER + j];
                }
            }

            // SiLU-SwiGLU activation grad.
            let mut d_h_gate = [0.0f32; INTER];
            let mut d_h_up = [0.0f32; INTER];
            for j in 0..INTER {
                d_h_gate[j] = d_act[j] * silu_grad(h_gate[j]) * h_up[j];
                d_h_up[j] = d_act[j] * silu(h_gate[j]);
            }

            // d_gate_w, d_up_w.
            for j in 0..INTER {
                for i in 0..HIDDEN {
                    d_gate_w[e][j * HIDDEN + i] += d_h_gate[j] * x_row[i];
                    d_up_w[e][j * HIDDEN + i] += d_h_up[j] * x_row[i];
                }
            }

            // d_x.
            for i in 0..HIDDEN {
                let mut acc = 0.0f32;
                for j in 0..INTER {
                    acc += gate_w[j * HIDDEN + i] * d_h_gate[j] + up_w[j * HIDDEN + i] * d_h_up[j];
                }
                d_x[t * HIDDEN + i] += acc;
            }
        }
    }

    MoEGrads {
        d_x,
        d_gate_w,
        d_up_w,
        d_down_w,
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Flatten per-expert weight arrays into the device layout [num_experts, R*C].
fn flatten_experts(weights: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for e in weights {
        out.extend_from_slice(e);
    }
    out
}

/// RMS relative error between device and host reference.
fn rms_rel_err(device: &[f32], host: &[f32]) -> f32 {
    assert_eq!(
        device.len(),
        host.len(),
        "length mismatch: device={} host={}",
        device.len(),
        host.len()
    );
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for (&d, &h) in device.iter().zip(host.iter()) {
        let denom = (h.abs() as f64).max(1e-6);
        let rel = (d - h) as f64 / denom;
        sum_sq += rel * rel;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt() as f32
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn p2_charon_backward_device_verifier() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_GPU_TEST unset or no ROCm device; skipping P2 device verifier");
        return;
    };

    let fw = flat_expert_weights();

    // Build oracle and get routing assignment.
    let moe = build_oracle(&fw);
    let x_tensor = cpu_tensor(fw.x.clone(), Shape::new(vec![BATCH, HIDDEN]));
    let (indices, weights) = moe.router.route(&x_tensor).expect("route");
    let assignment =
        grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(&indices, &weights)
            .expect("from_route");

    // Flatten expert weights into device layout.
    let gw_flat = flatten_experts(&fw.gate);
    let uw_flat = flatten_experts(&fw.up);
    let dw_flat = flatten_experts(&fw.down);
    let dy = vec![1.0f32; BATCH * HIDDEN];

    // Launch device backward.
    let result = dev
        .charon_grouped_backward_roundtrip(
            &fw.x,
            &gw_flat,
            &uw_flat,
            &dw_flat,
            &dy,
            &assignment,
            BATCH,
            HIDDEN,
            INTER,
            RSF,
        )
        .expect("charon_grouped_backward_roundtrip");

    // Host analytical backward reference.
    let host = analytical_backward(&fw);

    // Assert non-trivial reference (at least one entry is nonzero per grad).
    let dgw_max: f32 = host
        .d_gate_w
        .iter()
        .flatten()
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max);
    let duw_max: f32 = host
        .d_up_w
        .iter()
        .flatten()
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max);
    let ddw_max: f32 = host
        .d_down_w
        .iter()
        .flatten()
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max);
    let dx_max: f32 = host.d_x.iter().copied().map(f32::abs).fold(0.0, f32::max);
    assert!(
        dgw_max > 1e-6,
        "host d_gate_w is all-zero — reference degenerate"
    );
    assert!(
        duw_max > 1e-6,
        "host d_up_w is all-zero — reference degenerate"
    );
    assert!(
        ddw_max > 1e-6,
        "host d_down_w is all-zero — reference degenerate"
    );
    assert!(dx_max > 1e-6, "host d_x is all-zero — reference degenerate");

    // Compare device vs host per grad buffer.
    // The device layout is [num_experts, inter*hidden] (gate_w, up_w),
    // [num_experts, hidden*inter] (down_w), [batch, hidden] (d_x).
    let host_dgw_flat = flatten_experts(&host.d_gate_w);
    let host_duw_flat = flatten_experts(&host.d_up_w);
    let host_ddw_flat = flatten_experts(&host.d_down_w);

    let tol = 0.01f32; // generous for f32 atomicAdd + recompute; bugs surface at >> 0.1

    let dgw_err = rms_rel_err(&result.d_gate_w, &host_dgw_flat);
    eprintln!("d_gate_w: rms_rel_err = {dgw_err:.6} (tol={tol})");
    assert!(
        dgw_err <= tol,
        "P2 d_gate_w: RMS rel err {dgw_err} exceeds {tol}"
    );

    let duw_err = rms_rel_err(&result.d_up_w, &host_duw_flat);
    eprintln!("d_up_w: rms_rel_err = {duw_err:.6} (tol={tol})");
    assert!(
        duw_err <= tol,
        "P2 d_up_w: RMS rel err {duw_err} exceeds {tol}"
    );

    let ddw_err = rms_rel_err(&result.d_down_w, &host_ddw_flat);
    eprintln!("d_down_w: rms_rel_err = {ddw_err:.6} (tol={tol})");
    assert!(
        ddw_err <= tol,
        "P2 d_down_w: RMS rel err {ddw_err} exceeds {tol}"
    );

    let dx_err = rms_rel_err(&result.d_x, &host.d_x);
    eprintln!("d_x: rms_rel_err = {dx_err:.6} (tol={tol})");
    assert!(dx_err <= tol, "P2 d_x: RMS rel err {dx_err} exceeds {tol}");
}
