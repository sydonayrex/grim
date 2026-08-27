//! WI-F3 — WMMA-backed grouped MoE compute wiring tests.
//!
//! RED-first per the fusion-boundary plan:
//! * `wmma_grouped_dispatch_matches_moe_ffn_oracle` — the WMMA grouped
//!   dispatch path (`grim_moe_fused_grouped_wmma`) must match the real
//!   `MoeFfn::forward` CPU oracle on a synthetic top-k>1, multi-expert case,
//!   with the oracle-integrity precondition (RSF scaling) enforced as an
//!   actual call at the top of the test body, not a documentation note.
//! * `selector_routes_dispatch_target_not_just_enum` — `CharonSelector`
//!   output must route to the WMMA kernel entry only for
//!   `LargeGroupPrefill`, and to the scalar grouped kernel otherwise —
//!   asserting the dispatch *target*, so a regression that picks the right
//!   enum but calls the wrong kernel is caught.

use std::panic;

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::kernels::charon::{
    CharonSelector, CharonVariant, default_variant_table, grouped_dispatch_entry,
};
use grim_nn::Linear;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::shape::Shape;

const HIDDEN: usize = 8;
const INTER: usize = 8;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const BATCH: usize = 4;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

// ── deterministic synthetic MoE (same generators as golden_charon_moe_gpu) ──

// The three-tensor return mirrors the GPU weight layout exactly.
#[allow(clippy::type_complexity)]
fn deterministic_expert_weights() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut gate = Vec::with_capacity(NUM_EXPERTS);
    let mut up = Vec::with_capacity(NUM_EXPERTS);
    let mut down = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        let seed = (e + 1) as f32;
        let mut g = vec![0.0f32; INTER * HIDDEN];
        let mut u = vec![0.0f32; INTER * HIDDEN];
        for j in 0..INTER {
            for i in 0..HIDDEN {
                let idx = j * HIDDEN + i;
                g[idx] = ((i as f32 + j as f32 + seed) * 0.3).sin() * 0.5;
                u[idx] = ((i as f32 - j as f32 + seed) * 0.2).cos() * 0.5 + 0.5;
            }
        }
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
    (gate, up, down)
}

fn deterministic_router_gate() -> Vec<f32> {
    let mut gw = vec![0.0f32; NUM_EXPERTS * HIDDEN];
    for e in 0..NUM_EXPERTS {
        for i in 0..HIDDEN {
            gw[e * HIDDEN + i] = ((e as f32 + 1.0) * 0.5 + i as f32 * 0.1).sin();
        }
    }
    gw
}

fn deterministic_activations() -> Vec<f32> {
    let mut x = vec![0.0f32; BATCH * HIDDEN];
    for t in 0..BATCH {
        for i in 0..HIDDEN {
            x[t * HIDDEN + i] = ((t as f32 + 1.0) * 0.7 + i as f32 * 0.3).sin();
        }
    }
    x
}

fn build_moe_oracle(routed_scaling_factor: f32) -> MoeFfn {
    let (gate, up, down) = deterministic_expert_weights();
    let rgate = Linear::from_tensor(
        cpu_tensor(
            deterministic_router_gate(),
            Shape::new(vec![NUM_EXPERTS, HIDDEN]),
        ),
        None,
    );
    let mut eg = Vec::with_capacity(NUM_EXPERTS);
    let mut eu = Vec::with_capacity(NUM_EXPERTS);
    let mut ed = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        eg.push(Linear::from_tensor(
            cpu_tensor(gate[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        eu.push(Linear::from_tensor(
            cpu_tensor(up[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        ed.push(Linear::from_tensor(
            cpu_tensor(down[e].clone(), Shape::new(vec![HIDDEN, INTER])),
            None,
        ));
    }
    let bank = ExpertBank::from_linears(eg, eu, ed);
    let router = MoeRouter::new(rgate, RouterKind::SoftmaxTopK, TOP_K, NUM_EXPERTS, None);
    MoeFfn::new(router, bank, None, routed_scaling_factor)
}

fn flatten_expert_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (gate, up, down) = deterministic_expert_weights();
    let mut gf = Vec::with_capacity(NUM_EXPERTS * INTER * HIDDEN);
    let mut uf = Vec::with_capacity(NUM_EXPERTS * INTER * HIDDEN);
    let mut df = Vec::with_capacity(NUM_EXPERTS * HIDDEN * INTER);
    for e in 0..NUM_EXPERTS {
        gf.extend_from_slice(&gate[e]);
        uf.extend_from_slice(&up[e]);
        df.extend_from_slice(&down[e]);
    }
    (gf, uf, df)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Oracle-integrity precondition (closes the G-A3 documentation-only gap):
/// the CPU `MoeFfn::forward` must respect `routed_scaling_factor` before it
/// is trusted as the parity reference. This is the property tested by
/// `routed_scaling_factor_scales_routed_not_shared` in `grim-nn/src/moe.rs`,
/// inlined so it runs regardless of test invocation order.
fn assert_oracle_respects_rsf(x: &[f32], rsf1: &MoeFfn, rsf05: &MoeFfn) {
    let xt = cpu_tensor(x.to_vec(), Shape::new(vec![BATCH, HIDDEN]));
    let out1 = rsf1
        .forward(&xt)
        .expect("oracle rsf=1.0")
        .to_vec_f32()
        .expect("vec");
    let out05 = rsf05
        .forward(&xt)
        .expect("oracle rsf=0.5")
        .to_vec_f32()
        .expect("vec");
    let mut max_dev = 0.0f32;
    for i in 0..out1.len() {
        max_dev = max_dev.max((out05[i] - 0.5 * out1[i]).abs());
    }
    assert!(
        max_dev <= 1e-5,
        "oracle-integrity: MoeFfn::forward does not respect routed_scaling_factor \
         (rsf=0.5 deviates from 0.5*rsf=1.0 by {max_dev:.2e} > 1e-5); \
         GPU parity would compare against a wrong reference"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn wmma_grouped_dispatch_matches_moe_ffn_oracle() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };

    let routed_scaling_factor = 1.7f32; // non-trivial: not 1.0, exercises rsf multiply
    let x_vec = deterministic_activations();
    let x = cpu_tensor(x_vec.clone(), Shape::new(vec![BATCH, HIDDEN]));

    // Hard precondition, executed inside the test body (WI-F3 RED spec):
    // fail fast if the oracle is broken before any GPU comparison.
    let rsf1 = build_moe_oracle(1.0);
    let rsf05 = build_moe_oracle(0.5);
    assert_oracle_respects_rsf(&x_vec, &rsf1, &rsf05);

    let moe = build_moe_oracle(routed_scaling_factor);
    let cpu_v = moe
        .forward(&x)
        .expect("oracle forward")
        .to_vec_f32()
        .expect("vec");

    let (indices, weights) = moe.router.route(&x).expect("router route");
    let assignment =
        grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(&indices, &weights)
            .expect("from_route");
    // top-k>1, multi-expert case confirmed by construction (TOP_K=2,
    // NUM_EXPERTS=4); assert the routing actually spans >1 expert.
    let distinct: std::collections::BTreeSet<_> = indices.iter().flatten().collect();
    assert!(
        distinct.len() > 1,
        "synthetic routing must span multiple experts"
    );

    let (gw, uw, dw) = flatten_expert_weights();
    let gpu_v = dev
        .charon_grouped_dispatch_wmma_roundtrip(
            &x_vec,
            &gw,
            &uw,
            &dw,
            &assignment,
            BATCH,
            HIDDEN,
            INTER,
            routed_scaling_factor,
        )
        .expect("wmma grouped roundtrip");

    assert_eq!(gpu_v.len(), cpu_v.len(), "output length must match");
    let diff = max_abs_diff(&gpu_v, &cpu_v);
    assert!(
        diff <= 1e-3,
        "WI-F3 parity: WMMA grouped dispatch vs MoeFfn::forward oracle max-abs-err {diff} exceeds 1e-3\ngpu: {gpu_v:?}\ncpu: {cpu_v:?}"
    );

    // Also must match the scalar grouped kernel path (same routing/weights).
    let scalar_v = dev
        .charon_grouped_dispatch_roundtrip(
            &x_vec,
            &gw,
            &uw,
            &dw,
            &assignment,
            BATCH,
            HIDDEN,
            INTER,
            routed_scaling_factor,
        )
        .expect("scalar grouped roundtrip");
    let diff2 = max_abs_diff(&gpu_v, &scalar_v);
    assert!(
        diff2 <= 1e-3,
        "WMMA vs scalar grouped max-abs-err {diff2} exceeds 1e-3"
    );
}

/// WI-F3 gate 3 — the selector routes the *dispatch target*, not just the
/// enum: `LargeGroupPrefill` must resolve to the WMMA grouped kernel entry,
/// and the decode/skew variants must keep the scalar grouped entry.
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn selector_routes_dispatch_target_not_just_enum() {
    // Pure dispatch-target resolution.
    assert_eq!(
        grouped_dispatch_entry(CharonVariant::LargeGroupPrefill),
        "grim_moe_fused_grouped_wmma",
        "LargeGroupPrefill must route to the WMMA grouped kernel"
    );
    assert_eq!(
        grouped_dispatch_entry(CharonVariant::SmallBatchDecode),
        "grim_moe_fused_grouped",
        "decode regime keeps the scalar grouped kernel"
    );
    assert_eq!(
        grouped_dispatch_entry(CharonVariant::HighSkew),
        "grim_moe_fused_grouped",
        "high-skew regime keeps the scalar grouped kernel"
    );

    // End-to-end through the selector: a mid-skew prefill-shaped profile
    // selects LargeGroupPrefill, and the resolved dispatch target is the
    // WMMA entry — a regression that picks the right enum but maps it to the
    // wrong kernel fails here.
    let mut sel = CharonSelector::new(default_variant_table(), 1);
    let v = sel.select(0.5, 16.0, 4096.0, 1e7, 0.2);
    assert_eq!(v, CharonVariant::LargeGroupPrefill);
    assert_eq!(grouped_dispatch_entry(v), "grim_moe_fused_grouped_wmma");

    // Decode-shaped profile keeps the scalar entry end-to-end.
    let mut sel = CharonSelector::new(default_variant_table(), 1);
    let v0 = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
    assert_eq!(v0, CharonVariant::SmallBatchDecode);
    assert_eq!(grouped_dispatch_entry(v0), "grim_moe_fused_grouped");
}
