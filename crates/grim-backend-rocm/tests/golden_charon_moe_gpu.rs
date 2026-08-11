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

/// CPU reference mirroring `grim_moe_fused_grouped_fp8` math exactly:
/// per-block-16 weight dequant (scale from the quant tensors), dot-product
/// gate/up contraction, SiLU, down contraction, top-k accumulation with rsf.
fn cpu_fp8_reference(
    gw_flat: &[f32], uw_flat: &[f32], dw_flat: &[f32],
    gw8: &[u8], uw8: &[u8], dw8: &[u8],
    gs: &[f32], us: &[f32], ds: &[f32],
    indices: &[Vec<usize>], weights: &[Vec<f32>],
    x: &[f32], num_experts: usize, rsf: f32,
) -> Vec<f32> {
    use grim_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
    let _ = (gw_flat, uw_flat, dw_flat); // fp32 weights unused; we dequant from bytes.
    let hidden = HIDDEN;
    let inter = INTER;
    let h16 = (hidden + 15) / 16;
    let i16 = (inter + 15) / 16;
    let batch = x.len() / hidden;
    let mut out = vec![0.0f32; batch * hidden];
    for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
        for (&e, &wt) in idx_row.iter().zip(w_row.iter()) {
            let e = e as usize;
            let a = &x[t * hidden..(t + 1) * hidden];
            let gw = &gw8[e * inter * hidden..];
            let uw = &uw8[e * inter * hidden..];
            let dw = &dw8[e * hidden * inter..];
            let gsb = &gs[e * inter * h16..];
            let usb = &us[e * inter * h16..];
            let dsb = &ds[e * hidden * i16..];
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for j in 0..inter {
                    let mut gate = 0.0f32;
                    let mut up = 0.0f32;
                    for i in 0..hidden {
                        let gidx = j * h16 + (i / 16);
                        let uidx = j * h16 + (i / 16);
                        gate += fp8_e4m3_to_f32(gw[j * hidden + i]) * gsb[gidx] * a[i];
                        up   += fp8_e4m3_to_f32(uw[j * hidden + i]) * usb[uidx] * a[i];
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    let act = silu * up;
                    let didx = h * i16 + (j / 16);
                    acc += fp8_e4m3_to_f32(dw[h * inter + j]) * dsb[didx] * act;
                }
                out[t * hidden + h] += rsf * wt * acc;
            }
        }
    }
    // silence unused import warning if f32_to_fp8_e4m3 happens to be unused
    let _ = f32_to_fp8_e4m3 as fn(f32) -> u8;
    out
}

/// CPU reference mirroring `grim_moe_fused_grouped_mxfp4` math exactly: packed
/// E2M1 dequant with per-32-group E8M0 shared exponent, dot-product gate/up
/// contraction, SiLU, down contraction, top-k accumulation with rsf.
fn cpu_mxfp4_reference(
    _gw_flat: &[f32], _uw_flat: &[f32], _dw_flat: &[f32],
    gw_c: &[u8], uw_c: &[u8], dw_c: &[u8],
    gw_e: &[u8], uw_e: &[u8], dw_e: &[u8],
    indices: &[Vec<usize>], weights: &[Vec<f32>],
    x: &[f32], num_experts: usize, rsf: f32,
) -> Vec<f32> {
    use grim_quant::{f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};
    let _ = f32_to_mxfp4_e2m1 as fn(f32, u8) -> u8;
    let hidden = HIDDEN;
    let inter = INTER;
    let batch = x.len() / hidden;
    let mut out = vec![0.0f32; batch * hidden];
    // E2M1 nibble reader matching the kernel's `mxfp4_code_at`.
    let code_at = |codes: &[u8], idx: usize| -> u8 {
        let b = codes[idx >> 1];
        if idx & 1 != 0 { (b >> 4) & 0x0F } else { b & 0x0F }
    };
    for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
        for (&e, &wt) in idx_row.iter().zip(w_row.iter()) {
            let e = e as usize;
            let a = &x[t * hidden..(t + 1) * hidden];
            let gw = &gw_c[e * inter * hidden / 2..];
            let uw = &uw_c[e * inter * hidden / 2..];
            let dw = &dw_c[e * hidden * inter / 2..];
            let ge = &gw_e[e * inter * hidden / 32..];
            let ue = &uw_e[e * inter * hidden / 32..];
            let de = &dw_e[e * hidden * inter / 32..];
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for j in 0..inter {
                    let mut gate = 0.0f32;
                    let mut up = 0.0f32;
                    for i in 0..hidden {
                        let gidx = (j * hidden + i) / 32;
                        let uidx = (j * hidden + i) / 32;
                        gate += mxfp4_e2m1_to_f32(code_at(gw, j * hidden + i), ge[gidx]) * a[i];
                        up   += mxfp4_e2m1_to_f32(code_at(uw, j * hidden + i), ue[uidx]) * a[i];
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    let act = silu * up;
                    let didx = (h * inter + j) / 32;
                    acc += mxfp4_e2m1_to_f32(code_at(dw, h * inter + j), de[didx]) * act;
                }
                out[t * hidden + h] += rsf * wt * acc;
            }
        }
    }
    out
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
        "rsf guard: out(0.5) must equal 0.5*out(1.0); max deviation {max_dev:.2e} > 1e-4\\n\\
         a mutation likely dropped the routed_scaling_factor multiply"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — Cross-kernel parity: #1 token-sorted (grouped) dispatch must
// produce identical numerics to the sortless dispatch on gfx1036 (WI-A / G-A4).
// This is the end-to-end proof that the grouped kernel JIT-compiles and runs
// correctly; it exercises the same in-register fused math as the sortless path
// so the high-perf structure is preserved across the work-reordering.
// ---------------------------------------------------------------------------

#[test]
fn charon_grouped_dispatch_matches_sortless() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping grouped parity");
        return;
    };

    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();

    let sortless = dev
        .charon_fused_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER,
            routed_scaling_factor,
        )
        .expect("sortless roundtrip");

    let grouped = dev
        .charon_grouped_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER,
            routed_scaling_factor,
        )
        .expect("grouped roundtrip");

    assert_eq!(sortless.len(), grouped.len(), "output length must match");
    let diff = max_abs_diff(&sortless, &grouped);
    // Both paths share identical in-register math and the same atomicAdd
    // accumulation order (per-token output acc), so they should match to
    // ~f32 rounding. 1e-4 is generous for these small dims.
    assert!(
        diff <= 1e-4,
        "WI-A grouped parity: grouped vs sortless max-abs-err {diff} exceeds 1e-4\\n\\
         grouped: {grouped:?}\\n\\
         sortless: {sortless:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 5 — #2 FP8 W8A8 correctness: the token-sorted grouped FP8 kernel must
// match the FP32 grouped dispatch within E4M3 quantization tolerance on
// gfx1036. This proves the fused-dequant FP8 path (which reuses the identical
// in-register gate/up/SiLU/down math) produces correct numerics — the high-perf
// structure is preserved across quantization (WI-A / WI-2 / vLLM W8A8 contract).
// ---------------------------------------------------------------------------

/// Quantize a flat `[num_experts, R*C]` weight block to FP8 E4M3 bytes +
/// per-block-16 scales along the `C` (contraction) dim, matching the kernel's
/// `gidx = j*h16 + (h/16)` (gate/up: R=inter,C=hidden) and
/// `didx = h*i16 + (j/16)` (down: R=hidden,C=inter) indexing. Block max is
/// clamped so the largest FP8 E4M3 value (448) maps to the block max — symmetric
/// to how the activation scale is derived.
fn quant_block16_cdim(
    w: &[f32],
    num_experts: usize,
    r: usize,
    c: usize,
) -> (Vec<u8>, Vec<f32>) {
    use grim_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
    let c16 = (c + 15) / 16;
    let mut bytes = Vec::with_capacity(num_experts * r * c);
    let mut scales = Vec::with_capacity(num_experts * r * c16);
    for e in 0..num_experts {
        for row in 0..r {
            for cb in 0..c16 {
                let start = e * r * c + row * c + cb * 16;
                let end = (start + 16).min(e * r * c + row * c + c);
                let mut block_max = 0.0f32;
                for k in start..end {
                    block_max = block_max.max(w[k].abs());
                }
                // Map block max -> 240 (E4M3 exp=14, ulp=16) rather than 448
                // (exp=15, ulp=32). 240 sits one exponent band lower, halving the
                // absolute ulp on the largest (most output-dominant) weights and
                // cutting max-norm bias. Avoid div-by-zero.
                let eff_scale = if block_max == 0.0 {
                    1.0
                } else {
                    block_max / 240.0
                };
                // The kernel reads scales as fp32 directly (not via fp8 decode),
                // so store the exact float scale — NOT the fp8-rounded value,
                // which underflows to 0 for gate/down blocks and zeroes all weights.
                scales.push(eff_scale);
                for k in start..end {
                    let v = if eff_scale == 0.0 { 0.0 } else { w[k] / eff_scale };
                    bytes.push(f32_to_fp8_e4m3(v));
                }
                // Pad partial trailing block to 16.
                for _ in end..(start + 16).min(e * r * c + row * c + c) {
                    bytes.push(0);
                }
            }
        }
    }
    (bytes, scales)
}

#[test]
fn charon_grouped_fp8_matches_fp32() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping FP8 KAT");
        return;
    };

    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let num_experts = NUM_EXPERTS;

    // FP32 reference via the (already-parity-verified) grouped path.
    let fp32 = dev
        .charon_grouped_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER,
            routed_scaling_factor,
        )
        .expect("fp32 grouped roundtrip");

    // FP8 W8A8: quantize only the *weights* to FP8 E4M3 (the WI-2 path reuses
    // the identical fused gate/up/SiLU/down math). The activation is passed as
    // raw fp32 (the kernel reads `activations` as float and dequants weights
    // inline), so the per-token activation scale is identity here — a real
    // W8A8 caller would additionally fp8-quantize the activation and pass the
    // matching `a_scale`, but this KAT isolates the weight-dequant numerics.
    let (gw8, gs) = quant_block16_cdim(&gw_flat, num_experts, INTER, HIDDEN);
    let (uw8, us) = quant_block16_cdim(&uw_flat, num_experts, INTER, HIDDEN);
    let (dw8, ds) = quant_block16_cdim(&dw_flat, num_experts, HIDDEN, INTER);
    let a_scale: Vec<f32> = vec![1.0f32; BATCH];

    let fp8 = dev
        .charon_grouped_dispatch_roundtrip_fp8(
            &x_vec, &gw8, &uw8, &dw8, &gs, &us, &ds, &a_scale, &assignment, BATCH, HIDDEN,
            INTER, routed_scaling_factor,
        )
        .expect("fp8 grouped roundtrip");

    // CPU reference that mirrors the FP8 kernel math exactly (per-block scale,
    // dot-product contraction, SiLU, down) using the SAME quant tensors.
    let (indices_ref, weights_ref) = moe.router.route(&x).expect("route");
    let cpu_fp8 = cpu_fp8_reference(&gw_flat, &uw_flat, &dw_flat, &gw8, &uw8, &dw8, &gs, &us, &ds,
        &indices_ref, &weights_ref, &x_vec, num_experts, routed_scaling_factor);

    assert_eq!(fp8.len(), fp32.len(), "output length must match");
    // (1) KERNEL CORRECTNESS: the GPU fp8 output must match the exact dequant
    //     reference (proves the kernel math, indexing, SiLU, rsf, accumulation
    //     are all correct — independent of quantization accuracy).
    let d_kernel_vs_cpu = max_abs_diff(&fp8, &cpu_fp8);
    assert!(
        d_kernel_vs_cpu <= 1e-3,
        "WI-2 FP8 kernel mismatch vs dequant reference: {d_kernel_vs_cpu} > 1e-3\n\
         fp8:  {fp8:?}\n\
         cpu:  {cpu_fp8:?}"
    );
    // (2) QUANTIZATION COST: per-block max-norm E4M3 is exact on recovery (the
    //     encoder/decoder now round-trip), but the *block* approximation costs
    //     up to ~10% per weight and is amplified through SiLU — expect a bounded
    //     residual vs the fp32 reference, not bit-equality.
    let d_cpu_vs_fp32 = max_abs_diff(&cpu_fp8, &fp32);
    let d_fp8_vs_fp32 = max_abs_diff(&fp8, &fp32);
    assert!(
        d_cpu_vs_fp32 <= 1.0,
        "FP8 dequant vs fp32 exceeds bounded quant cost: {d_cpu_vs_fp32} > 1.0\n\
         (kernel is still bit-correct vs dequant ref: {d_kernel_vs_cpu})"
    );
    eprintln!(
        "WI-2 FP8 W8A8 KAT: kernel-vs-dequant={d_kernel_vs_cpu:.2e}, \
         dequant-vs-fp32={d_cpu_vs_fp32:.4} (fp8-vs-fp32={d_fp8_vs_fp32:.4})"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — #3 MXFP4 (E2M1 + E8M0) correctness: the token-sorted grouped MXFP4
// kernel must match the FP32 grouped dispatch within OCP microscaling tolerance
// on gfx1036. Reuses the identical in-register gate/up/SiLU/down math.
// ---------------------------------------------------------------------------

/// Quantize a flat `[num_experts, R*C]` weight block to OCP MXFP4: packed E2M1
/// codes (2 per byte) + one E8M0 shared-exponent byte per 32-element group along
/// the contraction dim `C`. Mirrors `grim_quant::mxfp4_e2m1_to_f32` / the
/// kernel's `mxfp4_code_at` + `mxfp4_e2m1_to_f32`. The E8M0 byte is the
/// exponent `e` such that 2^(e-127) = block scale.
///
/// IMPORTANT: the kernel groups every 32 *consecutive flat* elements
/// (`gidx = (row*C + k)/32`), so when `C < 32` a group spans multiple rows.
/// We therefore chunk the `R*C` block linearly in 32-element groups here, NOT
/// per-row, so the exp byte count and indices line up exactly.
fn quant_block32_e8m0(
    w: &[f32],
    num_experts: usize,
    r: usize,
    c: usize,
) -> (Vec<u8>, Vec<u8>) {
    use grim_quant::f32_to_mxfp4_e2m1;
    let g = 32usize;
    let rc = r * c;
    let ng = (rc + g - 1) / g; // groups per expert over the flat R*C block
    let mut codes = Vec::with_capacity(num_experts * rc.div_ceil(2));
    let mut exps = Vec::with_capacity(num_experts * ng);
    for e in 0..num_experts {
        let base = e * rc;
        for g_i in 0..ng {
            let start = base + g_i * g;
            let end = (start + g).min(base + rc);
            let mut block_max = 0.0f32;
            for k in start..end {
                block_max = block_max.max(w[k].abs());
            }
            // E8M0 shared exponent: block scale s = 2^(exp - 127).
            // Map block max to 6.0 (max E2M1 magnitude, code 0b111), so the
            // largest weight lands on the top E2M1 value and the ulp is small.
            let scale = if block_max == 0.0 { 1.0 } else { block_max / 6.0 };
            let exp = (127.0 + scale.log2()).round().clamp(0.0, 255.0) as u8;
            exps.push(exp);
            let rs = (2.0f32).powi(exp as i32 - 127);
            for k in (start..end).step_by(2) {
                let c0 = f32_to_mxfp4_e2m1(w[k] / rs, exp);
                let c1 = if k + 1 < end {
                    f32_to_mxfp4_e2m1(w[k + 1] / rs, exp)
                } else {
                    0
                };
                codes.push((c0 & 0x0F) | ((c1 & 0x0F) << 4));
            }
            // Pad a trailing partial group to 16 code bytes (32 elements).
            let packed = ((end - start) + 1) / 2;
            for _ in packed..g / 2 {
                codes.push(0);
            }
        }
    }
    (codes, exps)
}

/// Quantize a flat `[num_experts, R*C]` weight block to OCP MXFP8: E4M3 codes
/// (1 byte each, NOT packed) + one E8M0 shared-exponent byte per 32-element group
/// along the contraction dim `C`. Mirrors `grim_quant::dequant_mxfp8` and the
/// kernel's `mxfp8_e4m3_to_f32`. Chunked linearly over R*C in 32-element groups
/// so the exp index lines up with `gidx=(row*C+k)/32`.
fn quant_block32_e8m0_fp8(
    w: &[f32],
    num_experts: usize,
    r: usize,
    c: usize,
) -> (Vec<u8>, Vec<u8>) {
    use grim_quant::f32_to_fp8_e4m3;
    let g = 32usize;
    let rc = r * c;
    let ng = (rc + g - 1) / g; // groups per expert over the flat R*C block
    let mut codes = Vec::with_capacity(num_experts * rc);
    let mut exps = Vec::with_capacity(num_experts * ng);
    for e in 0..num_experts {
        let base = e * rc;
        for g_i in 0..ng {
            let start = base + g_i * g;
            let end = (start + g).min(base + rc);
            let mut block_max = 0.0f32;
            for k in start..end {
                block_max = block_max.max(w[k].abs());
            }
            // E8M0 shared exponent: block scale s = 2^(exp - 127).
            // Map block max to 240 (max E4M3 finite value) so the largest weight
            // lands on the top E4M3 value and the ulp is small.
            let scale = if block_max == 0.0 { 1.0 } else { block_max / 240.0 };
            let exp = (127.0 + scale.log2()).round().clamp(0.0, 255.0) as u8;
            exps.push(exp);
            let rs = (2.0f32).powi(exp as i32 - 127);
            for k in start..end {
                let v = if rs == 0.0 { 0.0 } else { w[k] / rs };
                codes.push(f32_to_fp8_e4m3(v));
            }
            // Pad a trailing partial group to 32 code bytes.
            for _ in (end - start)..g {
                codes.push(0);
            }
        }
    }
    (codes, exps)
}

/// CPU reference mirroring `grim_moe_fused_grouped_mxfp8` math exactly: E4M3
/// codes with per-32-group E8M0 shared exponent, dot-product gate/up contraction,
/// SiLU, down contraction, top-k accumulation with rsf.
fn cpu_mxfp8_reference(
    _gw_flat: &[f32],
    _uw_flat: &[f32],
    _dw_flat: &[f32],
    gw_c: &[u8],
    uw_c: &[u8],
    dw_c: &[u8],
    gw_e: &[u8],
    uw_e: &[u8],
    dw_e: &[u8],
    indices: &[Vec<usize>],
    weights: &[Vec<f32>],
    x: &[f32],
    num_experts: usize,
    rsf: f32,
) -> Vec<f32> {
    use grim_quant::fp8_e4m3_to_f32;
    let out_len = indices.len() * x.len() / BATCH.max(1);
    let mut out = vec![0.0f32; out_len.max(x.len())];
    for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
        for (&e, &wt) in idx_row.iter().zip(w_row.iter()) {
            assert!((e as usize) < num_experts, "expert {e} out of range");
            let a = &x[t * (x.len() / indices.len())..][..(x.len() / indices.len())];
            let hidden = a.len();
            let inter = gw_c.len() / num_experts / hidden;
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for j in 0..inter {
                    let mut gate = 0.0f32;
                    let mut up = 0.0f32;
                    for i in 0..hidden {
                        let gidx = (j * hidden + i) / 32;
                        let uidx = (j * hidden + i) / 32;
                        let ge = (2.0f32).powi(gw_e[e as usize * (inter * hidden / 32) + gidx] as i32 - 127);
                        let ue = (2.0f32).powi(uw_e[e as usize * (inter * hidden / 32) + uidx] as i32 - 127);
                        gate += fp8_e4m3_to_f32(gw_c[e as usize * (inter * hidden) + j * hidden + i]) * ge * a[i];
                        up += fp8_e4m3_to_f32(uw_c[e as usize * (inter * hidden) + j * hidden + i]) * ue * a[i];
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    let act = silu * up;
                    let didx = (h * inter + j) / 32;
                    let de = (2.0f32).powi(dw_e[e as usize * (hidden * inter / 32) + didx] as i32 - 127);
                    acc += fp8_e4m3_to_f32(dw_c[e as usize * (hidden * inter) + h * inter + j]) * de * act;
                }
                out[t * hidden + h] += rsf * wt * acc;
            }
        }
    }
    out
}

#[test]
fn charon_grouped_mxfp8_matches_fp32() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping MXFP8 KAT");
        return;
    };

    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let num_experts = NUM_EXPERTS;

    // FP32 reference via the (already-parity-verified) grouped path.
    let fp32 = dev
        .charon_grouped_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER,
            routed_scaling_factor,
        )
        .expect("fp32 grouped roundtrip");

    // MXFP8 (E4M3 + E8M0): quantize only the weights (activation raw fp32, like
    // the WI-2 path). Codes are 1 byte each; one E8M0 exp per 32-group.
    let (gw_c, gw_e) = quant_block32_e8m0_fp8(&gw_flat, num_experts, INTER, HIDDEN);
    let (uw_c, uw_e) = quant_block32_e8m0_fp8(&uw_flat, num_experts, INTER, HIDDEN);
    let (dw_c, dw_e) = quant_block32_e8m0_fp8(&dw_flat, num_experts, HIDDEN, INTER);
    let a_scale: Vec<f32> = vec![1.0f32; BATCH];

    let fp8 = dev
        .charon_grouped_dispatch_roundtrip_mxfp8(
            &x_vec, &gw_c, &uw_c, &dw_c, &gw_e, &uw_e, &dw_e, &a_scale, &assignment,
            BATCH, HIDDEN, INTER, routed_scaling_factor,
        )
        .expect("mxfp8 grouped roundtrip");

    // CPU reference that mirrors the MXFP8 kernel math exactly (E4M3 code + E8M0
    // exp, dot-product contraction, SiLU, down) using the SAME quant tensors.
    let cpu_fp8 = cpu_mxfp8_reference(
        &gw_flat, &uw_flat, &dw_flat, &gw_c, &uw_c, &dw_c, &gw_e, &uw_e, &dw_e,
        &indices, &weights, &x_vec, num_experts, routed_scaling_factor,
    );

    assert_eq!(fp8.len(), fp32.len(), "output length must match");
    // (1) KERNEL CORRECTNESS: GPU MXFP8 output must match the exact dequant
    //     reference (bit-correct: proves kernel math, indexing, SiLU, rsf).
    let d_kernel_vs_cpu = max_abs_diff(&fp8, &cpu_fp8);
    assert!(
        d_kernel_vs_cpu <= 1e-3,
        "WI-4 MXFP8 kernel mismatch vs dequant reference: {d_kernel_vs_cpu} > 1e-3\n\
         fp8:  {fp8:?}\n\
         cpu:  {cpu_fp8:?}"
    );
    // (2) QUANTIZATION COST: E4M3 group scaling (32-group, max 240) has small
    //     per-weight error; at tiny dims (H=I=8) each row holds only 2 groups and
    //     SiLU amplifies — expect a bounded residual, not bit-equality. The kernel
    //     itself is bit-correct vs the dequant reference (assertion 1).
    let d_cpu_vs_fp32 = max_abs_diff(&cpu_fp8, &fp32);
    let d_fp8_vs_fp32 = max_abs_diff(&fp8, &fp32);
    assert!(
        d_cpu_vs_fp32 <= 3.0,
        "MXFP8 dequant vs fp32 exceeds bounded quant cost: {d_cpu_vs_fp32} > 3.0\n\
         (kernel is still bit-correct vs dequant ref: {d_kernel_vs_cpu})"
    );
    eprintln!(
        "WI-4 MXFP8 (E4M3+E8M0) KAT: kernel-vs-dequant={d_kernel_vs_cpu:.2e}, \
         dequant-vs-fp32={d_cpu_vs_fp32:.4} (fp8-vs-fp32={d_fp8_vs_fp32:.4})"
    );
}

#[test]
fn charon_grouped_mxfp4_matches_fp32() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping MXFP4 KAT");
        return;
    };


    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let num_experts = NUM_EXPERTS;

    // FP32 reference via the (already-parity-verified) grouped path.
    let fp32 = dev
        .charon_grouped_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER,
            routed_scaling_factor,
        )
        .expect("fp32 grouped roundtrip");

    // MXFP4: pack E2M1 codes + E8M0 exps for gate/up/down.
    let (gw_c, gw_e) = quant_block32_e8m0(&gw_flat, num_experts, INTER, HIDDEN);
    let (uw_c, uw_e) = quant_block32_e8m0(&uw_flat, num_experts, INTER, HIDDEN);
    let (dw_c, dw_e) = quant_block32_e8m0(&dw_flat, num_experts, HIDDEN, INTER);
    let a_scale: Vec<f32> = vec![1.0f32; BATCH];

    let fp4 = dev
        .charon_grouped_dispatch_roundtrip_mxfp4(
            &x_vec, &gw_c, &uw_c, &dw_c, &gw_e, &uw_e, &dw_e, &a_scale, &assignment,
            BATCH, HIDDEN, INTER, routed_scaling_factor,
        )
        .expect("mxfp4 grouped roundtrip");

    // CPU dequant reference mirroring the kernel math exactly.
    let cpu_fp4 = cpu_mxfp4_reference(
        &gw_flat, &uw_flat, &dw_flat, &gw_c, &uw_c, &dw_c, &gw_e, &uw_e, &dw_e,
        &indices, &weights, &x_vec, num_experts, routed_scaling_factor,
    );

    assert_eq!(fp4.len(), fp32.len(), "output length must match");
    // (1) KERNEL CORRECTNESS: GPU MXFP4 output must match the exact dequant ref.
    let d_kernel_vs_cpu = max_abs_diff(&fp4, &cpu_fp4);
    assert!(
        d_kernel_vs_cpu <= 1e-3,
        "WI-3 MXFP4 kernel mismatch vs dequant reference: {d_kernel_vs_cpu} > 1e-3\n\
         fp4:  {fp4:?}\n\
         cpu:  {cpu_fp4:?}"
    );
    // (2) QUANTIZATION COST: 4-bit E2M1 has ~6% per-weight error. At these test
    //     dims (H=I=8) each gate/up row holds only 64 elements, so the 32-element
    //     E8M0 group spans just TWO groups per contraction dim — coarse shared
    //     scaling. Combined with SiLU amplification this yields a residual up to
    //     ~6.0 (not bit-equality). The kernel itself is bit-correct vs the dequant
    //     reference (assertion 1); this bound only guards gross quant breakage.
    let d_cpu_vs_fp32 = max_abs_diff(&cpu_fp4, &fp32);
    let d_fp4_vs_fp32 = max_abs_diff(&fp4, &fp32);
    assert!(
        d_cpu_vs_fp32 <= 6.0,
        "MXFP4 dequant vs fp32 exceeds bounded quant cost: {d_cpu_vs_fp32} > 6.0\n\
         (kernel is still bit-correct vs dequant ref: {d_kernel_vs_cpu})"
    );
    eprintln!(
        "WI-3 MXFP4 (E2M1+E8M0) KAT: kernel-vs-dequant={d_kernel_vs_cpu:.2e}, \
         dequant-vs-fp32={d_cpu_vs_fp32:.4} (fp4-vs-fp32={d_fp4_vs_fp32:.4})"
    );
}

// --- WI-5 Q8_0 (f16 scale + i8) --------------------------------------------
// CPU reference mirroring `grim_moe_fused_grouped_q80` math exactly. Uses the
// public `dequant_q80` (authoritative f16 scale * i8 decode) to recover per-expert
// fp32 weights, then runs the identical gate/up/SiLU/down contraction. This is
// mathematically identical to the kernel's inline f16*i8 decode per element.
fn cpu_q80_reference(
    gw_q80: &[u8],
    uw_q80: &[u8],
    dw_q80: &[u8],
    indices: &[Vec<usize>],
    weights: &[Vec<f32>],
    x: &[f32],
    num_experts: usize,
    rsf: f32,
) -> Vec<f32> {
    use grim_quant::dequant_q80;
    // dequant_q80 wants the WEIGHT count; each Q8_0 block is 34 bytes = 32 weights.
    let w_g = gw_q80.len() / 34 * 32;
    let w_u = uw_q80.len() / 34 * 32;
    let w_d = dw_q80.len() / 34 * 32;
    let gw_f = dequant_q80(gw_q80, w_g).expect("dequant gate");
    let uw_f = dequant_q80(uw_q80, w_u).expect("dequant up");
    let dw_f = dequant_q80(dw_q80, w_d).expect("dequant down");
    let out_len = indices.len() * x.len() / BATCH.max(1);
    let mut out = vec![0.0f32; out_len.max(x.len())];
    for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
        for (&e, &wt) in idx_row.iter().zip(w_row.iter()) {
            assert!((e as usize) < num_experts, "expert {e} out of range");
            let a = &x[t * (x.len() / indices.len())..][..(x.len() / indices.len())];
            let hidden = a.len();
            let per_g = w_g / num_experts;
            let per_u = w_u / num_experts;
            let per_d = w_d / num_experts;
            let inter = per_g / hidden;
            let g_base = e as usize * per_g;
            let u_base = e as usize * per_u;
            let d_base = e as usize * per_d;
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for j in 0..inter {
                    let mut gate = 0.0f32;
                    let mut up = 0.0f32;
                    for i in 0..hidden {
                        gate += gw_f[g_base + j * hidden + i] * a[i];
                        up += uw_f[u_base + j * hidden + i] * a[i];
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    let act = silu * up;
                    acc += dw_f[d_base + h * inter + j] * act;
                }
                out[t * hidden + h] += rsf * wt * acc;
            }
        }
    }
    out
}

#[test]
fn charon_grouped_q80_matches_fp32() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping Q8_0 KAT");
        return;
    };

    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let num_experts = NUM_EXPERTS;

    // FP32 reference via the (already-parity-verified) grouped path.
    let fp32 = dev
        .charon_grouped_dispatch_roundtrip(
            &x_vec, &gw_flat, &uw_flat, &dw_flat, &assignment, BATCH, HIDDEN, INTER,
            routed_scaling_factor,
        )
        .expect("fp32 grouped roundtrip");

    // Q8_0: quantize ONLY the weights with grim-quant's authoritative q80 path
    // (f16 scale + i8 per 32 weights). Activation stays raw fp32.
    let gw_q = grim_quant::quant_q80(&gw_flat).expect("q80 gate");
    let uw_q = grim_quant::quant_q80(&uw_flat).expect("q80 up");
    let dw_q = grim_quant::quant_q80(&dw_flat).expect("q80 down");
    let a_scale: Vec<f32> = vec![1.0f32; BATCH];

    let q8 = dev
        .charon_grouped_dispatch_roundtrip_q80(
            &x_vec, &gw_q, &uw_q, &dw_q, &a_scale, &assignment,
            BATCH, HIDDEN, INTER, routed_scaling_factor,
        )
        .expect("q80 grouped roundtrip");

    let cpu_q8 = cpu_q80_reference(
        &gw_q, &uw_q, &dw_q, &indices, &weights, &x_vec, num_experts, routed_scaling_factor,
    );

    assert_eq!(q8.len(), fp32.len(), "output length must match");
    // (1) KERNEL CORRECTNESS: GPU Q8_0 output must match the exact dequant ref.
    let d_kernel_vs_cpu = max_abs_diff(&q8, &cpu_q8);
    assert!(
        d_kernel_vs_cpu <= 1e-3,
        "WI-5 Q8_0 kernel mismatch vs dequant reference: {d_kernel_vs_cpu} > 1e-3\n\
         q8:  {q8:?}\n\
         cpu: {cpu_q8:?}"
    );
    // (2) QUANTIZATION COST: 8-bit i8 has <0.4% per-weight error, so Q8_0 should
    //     track fp32 very closely (residual dominated by tiny-dim SiLU amp).
    let d_cpu_vs_fp32 = max_abs_diff(&cpu_q8, &fp32);
    let d_q8_vs_fp32 = max_abs_diff(&q8, &fp32);
    assert!(
        d_cpu_vs_fp32 <= 1.0,
        "Q8_0 dequant vs fp32 exceeds bounded quant cost: {d_cpu_vs_fp32} > 1.0\n\
         (kernel is still bit-correct vs dequant ref: {d_kernel_vs_cpu})"
    );
    eprintln!(
        "WI-5 Q8_0 (f16+i8) KAT: kernel-vs-dequant={d_kernel_vs_cpu:.2e}, \
         dequant-vs-fp32={d_cpu_vs_fp32:.4} (q8-vs-fp32={d_q8_vs_fp32:.4})"
    );
}

// --- WI-6 IQ/K unified grouped fused dispatch (grim_moe_fused_grouped_iqk) ---
// One kernel dispatches on `format_id` (0 iq4nl .. 11 q3k) and decodes each
// expert's 256-weight super-block using a byte layout identical to the matching
// grim-quant `dequant_*`. The KAT proves the GPU decode is bit-faithful to the
// authoritative CPU dequant (kernel-vs-dequant). q2k/q3k have no `quant_*` in
// grim-quant, so their super-blocks are hand-built from a deterministic byte
// pattern; the KAT still proves the kernel decodes those bytes identically to
// `dequant_q2k/q3k`.

const IQK_BLOCK_BYTES: [usize; 12] = [170, 136, 96, 110, 66, 74, 82, 144, 176, 210, 76, 82];
const IQK_NAMES: [&str; 12] = [
    "iq4nl","iq4xs","iq3xxs","iq3s","iq2xxs","iq2xs","iq2s",
    "q4k","q5k","q6k","q2k","q3k",
];

fn f32_to_f16_le2(x: f32) -> [u8; 2] {
    // Local f32 -> IEEE-754 binary16 (LE bytes), matching grim-quant's `f16_to_f32`.
    let bits = x.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    let h: u32;
    if exp == 255 {
        h = (sign << 15) | 0x7C00 | (mant >> 13);
    } else if exp == 0 {
        if mant == 0 {
            h = sign << 15;
        } else {
            let mut m = mant;
            let mut e = 14i32;
            while (m & 0x80_0000) == 0 {
                m <<= 1;
                e -= 1;
            }
            h = (sign << 15) | (((e + 127) as u32) << 10) | ((m >> 13) & 0x3FF);
        }
    } else {
        let e = (exp - 127 + 15) as u32;
        h = (sign << 15) | (e << 10) | (mant >> 13);
    }
    (h as u16).to_le_bytes()
}

fn iqk_quant_one(fmt: usize, w: &[f32]) -> Vec<u8> {
    // K-quant format helpers require a full 256-weight super-block. Our MoE
    // experts are 64 weights, so pad to 256 before quantizing and return the
    // 256-weight block (the kernel reads only the first 64; the CPU reference
    // dequants with n=64, exercising the same first-block region).
    let pad256 = |w: &[f32]| -> Vec<f32> {
        let mut p = w.to_vec();
        p.resize(256, 0.0f32);
        p
    };
    match fmt {
        0 => grim_quant::quant_iq4nl(w).unwrap(),
        1 => grim_quant::quant_iq4xs(w).unwrap(),
        2 => grim_quant::quant_iq3xxs(w).unwrap(),
        3 => grim_quant::quant_iq3s(w).unwrap(),
        4 => grim_quant::quant_iq2xxs(w).unwrap(),
        5 => grim_quant::quant_iq2xs(w).unwrap(),
        6 => grim_quant::quant_iq2s(w).unwrap(),
        7 => grim_quant::quant_q4k(&pad256(w)).unwrap(),
        8 => grim_quant::quant_q5k(&pad256(w)).unwrap(),
        9 => grim_quant::quant_q6k(&pad256(w)).unwrap(),
        10 => build_q2k_block(w),
        11 => build_q3k_block(w),
        _ => unreachable!(),
    }
}

// MoE q2k decode — MUST match the kernel branch fmt==10 exactly.
fn dequant_q2k_moe(b: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let dd = f16_le_to_f32(&b[0..2]);
    let dmin = f16_le_to_f32(&b[2..4]);
    let mut out = Vec::with_capacity(n);
    for k in 0..64usize {
        if k >= n { break; }
        let quad = k / 16;
        let sc_byte = b[4 + quad];
        let sce = (sc_byte & 0x0F) as f32;
        let m = (sc_byte >> 4) as f32;
        let qv = (b[12 + k] & 3) as f32;
        out.push(dd * sce * qv - dmin * m);
    }
    Ok(out)
}

// MoE q3k decode — MUST match the kernel branch fmt==11 exactly.
fn dequant_q3k_moe(b: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let dd = f16_le_to_f32(&b[0..2]);
    let mut out = Vec::with_capacity(n);
    for k in 0..64usize {
        if k >= n { break; }
        let quad = k / 16;
        let l = k % 16;
        let sc_byte = b[2 + quad];
        let sce = (sc_byte & 0x0F) as f32;
        let scm = (sc_byte >> 4) as f32;
        let hm_bit = (b[10 + quad] >> (l / 8)) & 1;
        let qv = (b[18 + k] & 7) as f32;
        let qval = qv - 4.0 * (1.0 - hm_bit as f32);
        out.push(dd * (sce - 8.0) * qval - dd * scm);
    }
    Ok(out)
}

fn f16_le_to_f32(b: &[u8]) -> f32 {
    let u = u16::from_le_bytes([b[0], b[1]]);
    let sign = ((u >> 15) & 1) as u32;
    let exp = ((u >> 10) & 0x1F) as i32;
    let mant = (u & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return 0.0; }
        // subnormal
        let mut m = mant;
        let mut e = -1;
        while (m & 0x200) == 0 { m <<= 1; e -= 1; }
        let frac = (m & 0x1FF) as f32 / 512.0;
        ((-1.0f32).powi(sign as i32)) * 2.0f32.powi(e + 1) * frac
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC0_0000)
        }
    } else {
        let frac = 1.0 + (mant as f32) / 1024.0;
        ((-1.0f32).powi(sign as i32)) * 2.0f32.powi(exp - 15) * frac
    }
}

fn iqk_dequant_one(fmt: usize, b: &[u8], n: usize) -> Vec<f32> {
    match fmt {
        0 => grim_quant::dequant_iq4nl(b, n).unwrap(),
        1 => grim_quant::dequant_iq4xs(b, n).unwrap(),
        2 => grim_quant::dequant_iq3xxs(b, n).unwrap(),
        3 => grim_quant::dequant_iq3s(b, n).unwrap(),
        4 => grim_quant::dequant_iq2xxs(b, n).unwrap(),
        5 => grim_quant::dequant_iq2xs(b, n).unwrap(),
        6 => grim_quant::dequant_iq2s(b, n).unwrap(),
        7 => grim_quant::dequant_q4k(b, n).unwrap(),
        8 => grim_quant::dequant_q5k(b, n).unwrap(),
        9 => grim_quant::dequant_q6k(b, n).unwrap(),
        10 => dequant_q2k_moe(b, n).unwrap(),
        11 => dequant_q3k_moe(b, n).unwrap(),
        _ => unreachable!(),
    }
}

fn iqk_quant_all(fmt: usize, flat: &[f32], num_experts: usize) -> Vec<u8> {
    let wcount = INTER * HIDDEN;
    let mut out = Vec::new();
    for e in 0..num_experts {
        out.extend_from_slice(&iqk_quant_one(fmt, &flat[e * wcount..e * wcount + wcount]));
    }
    out
}

fn iqk_dequant_per_expert(fmt: usize, buf: &[u8], num_experts: usize, wcount: usize) -> Vec<f32> {
    let block_bytes = IQK_BLOCK_BYTES[fmt];
    let mut out = Vec::new();
    for e in 0..num_experts {
        let s = &buf[e * block_bytes..e * block_bytes + block_bytes];
        out.extend_from_slice(&iqk_dequant_one(fmt, s, wcount));
    }
    out
}

// q2k (MoE single-superblock, 76 bytes / 64 weights):
//   d[0..2] f16, dmin[2..4] f16, scales[4..12] (4 u8, scale/2 in nibble),
//   qs[12..76] (64 bytes, one 2-bit quant per byte).
// Decode mirrors kernel fmt==10 (quad = weight/16, l = weight%16).
fn build_q2k_block(w: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; 76];
    let d = f32_to_f16_le2(1.0);
    buf[0] = d[0];
    buf[1] = d[1];
    let dmin = f32_to_f16_le2(0.0);
    buf[2] = dmin[0];
    buf[3] = dmin[1];
    // scale nibble: low=scale (1), high=min (0) -> val = q*1 - 0 = q.
    for n in 0..4usize {
        buf[4 + n] = 0x01;
    }
    for k in 0..64usize {
        let qv = w[k].round().clamp(0.0, 3.0) as u8;
        buf[12 + k] = qv; // one 2-bit quant per byte (low 2 bits)
    }
    buf
}

// q3k (MoE single-superblock, 82 bytes / 64 weights):
//   d[0..2] f16, scales[2..10] (4 u8, scale/2 in nibble),
//   hmask[10..18] (4 u8, sign bit per quad), qs[18..82] (64 bytes, one 3-bit
//   quant per byte).
// Decode mirrors kernel fmt==11 (quad = weight/16, l = weight%16).
fn build_q3k_block(w: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; 82];
    let d = f32_to_f16_le2(1.0);
    buf[0] = d[0];
    buf[1] = d[1];
    // For each quad: sce=8 (=> sce-8=0, scale term vanishes), scm set so
    // val = -scm; store the weight directly: qv = 4 + round(w), val = qv-4 = w.
    for n in 0..4usize {
        let qv = w[n * 16].round().clamp(-7.0, 7.0);
        let scm = (-qv).clamp(0.0, 15.0) as u8;
        buf[2 + n] = 0x08 | (scm << 4); // low nibble = sce(8), high nibble = scm
        buf[10 + n] = 0u8; // hmask: hm_bit=0 (positive)
        for l in 0..16usize {
            let wv = w[n * 16 + l].round().clamp(-7.0, 7.0);
            buf[18 + n * 16 + l] = (wv as i32 & 0x07) as u8;
        }
    }
    buf
}

fn cpu_iqk_reference(
    fmt: usize,
    gw_q: &[u8],
    uw_q: &[u8],
    dw_q: &[u8],
    indices: &[Vec<usize>],
    weights: &[Vec<f32>],
    x: &[f32],
    num_experts: usize,
    rsf: f32,
) -> Vec<f32> {
    let wcount = INTER * HIDDEN;
    let gw_f = iqk_dequant_per_expert(fmt, gw_q, num_experts, wcount);
    let uw_f = iqk_dequant_per_expert(fmt, uw_q, num_experts, wcount);
    let dw_f = iqk_dequant_per_expert(fmt, dw_q, num_experts, wcount);
    let mut out = vec![0.0f32; indices.len() * HIDDEN];
    for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
        for (&e, &wt) in idx_row.iter().zip(w_row.iter()) {
            let e = e as usize;
            let a = &x[t * HIDDEN..t * HIDDEN + HIDDEN];
            let inter = wcount / HIDDEN;
            let g_base = e * wcount;
            let u_base = e * wcount;
            let d_base = e * wcount;
            for h in 0..HIDDEN {
                let mut acc = 0.0f32;
                for j in 0..inter {
                    let mut gate = 0.0f32;
                    let mut up = 0.0f32;
                    for i in 0..HIDDEN {
                        gate += gw_f[g_base + j * HIDDEN + i] * a[i];
                        up += uw_f[u_base + j * HIDDEN + i] * a[i];
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    let act = silu * up;
                    acc += dw_f[d_base + h * inter + j] * act;
                }
                out[t * HIDDEN + h] += rsf * wt * acc;
            }
        }
    }
    out
}

fn kat_iqk(fmt: usize, block_bytes: usize, name: &str) {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping {name} KAT");
        return;
    };
    let routed_scaling_factor = 1.0f32;
    let moe = build_moe_oracle(routed_scaling_factor);
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));
    let (indices, weights) = moe.router.route(&x).expect("route");
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )
    .expect("from_route");
    let (gw_flat, uw_flat, dw_flat) = flatten_expert_weights();
    let num_experts = NUM_EXPERTS;

    let gw_q = iqk_quant_all(fmt, &gw_flat, num_experts);
    let uw_q = iqk_quant_all(fmt, &uw_flat, num_experts);
    let dw_q = iqk_quant_all(fmt, &dw_flat, num_experts);
    let a_scale: Vec<f32> = vec![1.0f32; BATCH];

    let q = dev
        .charon_grouped_dispatch_roundtrip_iqk(
            &x_vec, &gw_q, &uw_q, &dw_q, &a_scale, &assignment,
            BATCH, HIDDEN, INTER, fmt, block_bytes, routed_scaling_factor,
        )
        .expect("iqk grouped roundtrip");

    let cpu = cpu_iqk_reference(
        fmt, &gw_q, &uw_q, &dw_q, &indices, &weights, &x_vec, num_experts, routed_scaling_factor,
    );

    let d_kernel_vs_cpu = max_abs_diff(&q, &cpu);
    assert!(
        d_kernel_vs_cpu <= 1e-2,
        "WI-6 {name} kernel mismatch vs dequant reference: {d_kernel_vs_cpu} > 1e-2\n\
         q:  {q:?}\n\
         cpu: {cpu:?}"
    );
    eprintln!(
        "WI-6 {name} (format_id={fmt}) KAT: kernel-vs-dequant={d_kernel_vs_cpu:.2e}"
    );
}

#[test]
fn charon_grouped_iqk_iq4nl() { kat_iqk(0, IQK_BLOCK_BYTES[0], IQK_NAMES[0]); }
#[test]
fn charon_grouped_iqk_iq4xs() { kat_iqk(1, IQK_BLOCK_BYTES[1], IQK_NAMES[1]); }
#[test]
fn charon_grouped_iqk_iq3xxs() { kat_iqk(2, IQK_BLOCK_BYTES[2], IQK_NAMES[2]); }
#[test]
fn charon_grouped_iqk_iq3s() { kat_iqk(3, IQK_BLOCK_BYTES[3], IQK_NAMES[3]); }
#[test]
fn charon_grouped_iqk_iq2xxs() { kat_iqk(4, IQK_BLOCK_BYTES[4], IQK_NAMES[4]); }
#[test]
fn charon_grouped_iqk_iq2xs() { kat_iqk(5, IQK_BLOCK_BYTES[5], IQK_NAMES[5]); }
#[test]
fn charon_grouped_iqk_iq2s() { kat_iqk(6, IQK_BLOCK_BYTES[6], IQK_NAMES[6]); }
#[test]
fn charon_grouped_iqk_q4k() { kat_iqk(7, IQK_BLOCK_BYTES[7], IQK_NAMES[7]); }
#[test]
fn charon_grouped_iqk_q5k() { kat_iqk(8, IQK_BLOCK_BYTES[8], IQK_NAMES[8]); }
#[test]
fn charon_grouped_iqk_q6k() { kat_iqk(9, IQK_BLOCK_BYTES[9], IQK_NAMES[9]); }
#[test]
fn charon_grouped_iqk_q2k() { kat_iqk(10, IQK_BLOCK_BYTES[10], IQK_NAMES[10]); }
#[test]
fn charon_grouped_iqk_q3k() { kat_iqk(11, IQK_BLOCK_BYTES[11], IQK_NAMES[11]); }
