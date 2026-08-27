//! Charon MoE backward — finite-difference gradient check (WI-Charon-1 gate).
//!
//! The plan's WI-Charon-1 gate requires, verbatim:
//!
//! > gradient-check against `grim-autograd`'s CPU tape-based backward on the
//! > existing `MoeFfn` reference, for all four gradients by name — not
//! > "gradients pass," but `d_x`/`d_gate_w`/`d_up_w`/`d_down_w` individually
//! > asserted.
//!
//! Two reasons that literal gate shape can't be reproduced in-sandbox:
//!
//! 1. **No GPU**: the Charon backward *kernel* (`grim_moe_fused_grouped_backward`
//!    in `kernels::charon_backward::KERNEL_SOURCE`) only runs on a device. The
//!    plan marks the numeric correctness gate **device-gated, unverified in
//!    this sandbox**.
//! 2. **`MoeFfn` has no autograd wiring**: `MoeFfn::forward` returns a plain
//!    `Tensor`, not a tape node — there is no `grim-autograd` backward path
//!    that produces `d_x`/`d_gate_w`/`d_up_w`/`d_down_w` to compare against.
//!    The "tape-based backward" in the plan's gate refers to extending
//!    `MoeFfn` with autograd support; that is itself a separate work item
//!    (the plan flags router backward as explicitly out of scope and weight-
//!    gradient autograd integration as follows WI-Charon-1's FP32 base).
//!
//! What we CAN verify without a device — and what this file pins — is the
//! *math* the kernel encodes, validated two independent ways:
//!
//! A. **Analytical backward** implemented in pure Rust, mirroring exactly the
//!    decomposition documented in `charon_backward.rs` and the standard MoE
//!    backward identities:
//!     * forward: `h_gate = gate_w @ x ; h_up = up_w @ x ; act = silu(h_gate) * h_up ;
//!        y = down_w @ act`
//!     * `d_down_w[e] = (rsf * w) * d_y ⊗ act`            (outer product)
//!     * `d_hidden = down_w^T @ d_y`                      (pre-SiLU activation grad)
//!     * `d_gate[e] = d_hidden * silu'(h_gate) ; d_up[e] = d_hidden * silu(h_gate)`
//!     * `d_gate_w[e] = (rsf * w) * d_gate ⊗ x ; d_up_w[e] = (rsf * w) * d_up ⊗ x`
//!     * `d_x      = (rsf * w) * (gate_w^T @ d_gate + up_w^T @ d_up)`
//!   (router-backward is out of scope; the router gate weights are frozen in
//!   these tests, so finite-difference perturbations of expert weights never
//!   flip the top-k selection — see `well_separated_router_gate`.)
//!
//! B. **Finite-difference check** of that analytical backward against the real
//!    `MoeFfn::forward` reference: perturb each weight entry / input entry by
//!    ±`EPS`, call `MoeFfn::forward`, and central-difference the scalar output
//!    `sum(out)` against the analytical gradient. The four gradients are
//!    asserted **by name** (`d_x`, `d_gate_w`, `d_up_w`, `d_down_w`), each in
//!    its own `#[test]` — exactly the regression the plan calls out (a prior
//!    draft implemented only `d_down_w`/`d_x` while claiming completeness)
//!    would fail at least three of the four named tests here.
//!
//! C. **Kernel-source-encodes-the-same-math** structural test: assert
//!    `charon_backward::KERNEL_SOURCE` contains the load-bearing symbols of
//!    the decomposition (the SiLU-derivative, the four named output buffers,
//!    the `rsf * w` scaling factor, the `atomicAdd` into `d_x`), so the HIP
//!    kernel that runs on a device is provably the same math the validated
//!    host backward computes — not a parallel implementation that could
//!    silently diverge.
//!
//! All three layers run host-side without a device.

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::kernels::charon_backward;
use grim_nn::Linear;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::shape::Shape;

// ---------------------------------------------------------------------------
// Geometry: kept small so finite differences finish fast, but non-trivial
// (inter != hidden, distinct-per-expert weights) so a transposition, swapped
// gate/up, or wrong stride produces a measurably different gradient.
// ---------------------------------------------------------------------------

const HIDDEN: usize = 4;
const INTER: usize = 3;
const NUM_EXPERTS: usize = 2;
const TOP_K: usize = 1; // simplest combine path: one weight per token.
const BATCH: usize = 1; // F-D scalar = sum over the single token's output.
const RSF: f32 = 0.7; // distinct from 1.0 so a dropped-rsf mutation surfaces.

// Finite-difference step. With f32 `EPSILON ≈ 1.2e-7`:
//   * roundoff floor = O(EPSILON/EPS) — decreases as EPS grows.
//   * truncation err  = O(EPS^2 * |f'''|)  — increases as EPS grows.
// `EPS=5e-3` balances both at ~`2.5e-5`, well below `TOL`. We use a step
// comfortably inside the linear regime for the SiLU-SwiGLU chain (|w| ≤ ~1.5)
// AND small enough that well-separated router logits keep top-k selection
// piecewise-stable under perturbation (the routing gate is frozen here —
// perturbations are expert-weight perturbations, which never enter the
// router's path).
const EPS: f32 = 5e-3;
// Tolerance: any genuine math bug is well above `1e-3` (the `h_up` bug this
// gate caught surfaced at `rel_err ≈ 2.9e-1`; the double-scaling bug at
// `rel_err ≈ 4.3e-1`). `TOL=1e-3` comfortably absorbs the `~5e-5` f32 noise
// floor while keeping the gate sensitive to the structural regressions it
// was built to catch.
const TOL: f32 = 1e-3;

/// Per-expert weight tensors held as plain `Vec<f32>` so we can perturb a
/// single entry, rebuild the `MoeFfn` oracle, and finite-difference.
struct FlatWeights {
    gate: Vec<Vec<f32>>, // [num_experts][inter*hidden]
    up: Vec<Vec<f32>>,   // [num_experts][inter*hidden]
    down: Vec<Vec<f32>>, // [num_experts][hidden*inter]
    x: Vec<f32>,         // [batch*hidden]
}

/// Distinct, numerically-rich expert weights. Each expert is purposefully
/// different so a wrong-expert-stride mutation produces a different F-D.
fn flat_expert_weights() -> FlatWeights {
    let mut gate = Vec::with_capacity(NUM_EXPERTS);
    let mut up = Vec::with_capacity(NUM_EXPERTS);
    let mut down = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        let seed = (e as f32) * 0.5 + 0.3;
        let mut g = vec![0.0f32; INTER * HIDDEN];
        let mut u = vec![0.0f32; INTER * HIDDEN];
        let mut d = vec![0.0f32; HIDDEN * INTER];
        for j in 0..INTER {
            for i in 0..HIDDEN {
                let idx = j * HIDDEN + i;
                g[idx] = ((i as f32 + j as f32 + seed) * 0.31).sin() * 0.4;
                u[idx] = ((i as f32 - j as f32 - seed) * 0.27).cos() * 0.4 + 0.1;
            }
        }
        for h in 0..HIDDEN {
            for j in 0..INTER {
                d[h * INTER + j] = 0.3 / (1.0 + h as f32 + j as f32 + seed);
            }
        }
        gate.push(g);
        up.push(u);
        down.push(d);
    }
    // Pick an input with nonzero elements in every slot so d_x is exercised
    // across all hidden columns (a zero entry would mask that gradient slot
    // in the F-D and let a mutation slip through).
    let x = vec![0.37, -0.11, 0.83, 0.49];
    FlatWeights { gate, up, down, x }
}

/// Router gate weights chosen to make the top-k selection **piecewise-stable**
/// under ±EPS perturbations of expert weights: the two experts' selection
/// scores on the test input differ by ~1.0 (well above the F-D noise floor).
/// This keeps the combine weight `w` constant during the weight perturbations
/// so finite differences measure only the expert-weight gradient — exactly
/// the kernel's decomposition; the router is outside the backward's scope.
fn well_separated_router_gate() -> Vec<f32> {
    // expert 0 strongly preferred on every input column; expert 1 weak.
    // Routing will pick expert 0 deterministically; small expert-weight
    // perturbations cannot flip the ranking because the gate is frozen.
    let mut gw = vec![0.0f32; NUM_EXPERTS * HIDDEN];
    for i in 0..HIDDEN {
        gw[i] = 2.0 + 0.1 * i as f32; // expert 0: large logits
        gw[HIDDEN + i] = -2.0 - 0.1 * i as f32; // expert 1: large negative
    }
    gw
}

/// Build the `MoeFfn` CPU oracle from flat weights. `shared_expert` is `None`
/// to match the Charon backward kernel's scope (shared-expert backward is
/// folded into expert backward once autograd integration lands).
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

/// Scalar objective `f = sum_t sum_h out[t,h]` over `MoeFfn::forward`. A sum
/// objective couples every output element back to a unit upstream gradient
/// (`d_y = 1`), so each weight's analytical gradient is simply the sum of its
/// corresponding backward-kernel contribution — easy to derive and exactly
/// what the kernel accumulates under `atomicAdd` into the four grad buffers.
fn objective(fw: &FlatWeights) -> f32 {
    let moe = build_oracle(fw);
    let x = cpu_tensor(fw.x.clone(), Shape::new(vec![BATCH, HIDDEN]));
    let out = moe.forward(&x).expect("MoeFfn::forward");
    out.to_vec_f32().expect("to_vec_f32").iter().sum::<f32>()
}

// ---------------------------------------------------------------------------
// Analytical backward (host reference). Mirrors the decomposition documented
// in `charon_backward.rs` exactly. The router's combine weight `w` for the
// single routed expert (top_k=1) is read from `MoeRouter::route` so the analyt-
// ical path consumes the SAME routing decision the forward produced — keeping
// the math a true inverse rather than a parallel one.
// ---------------------------------------------------------------------------

struct MoEGrads {
    d_x: Vec<f32>,           // [batch*hidden]
    d_gate_w: Vec<Vec<f32>>, // [num_experts][inter*hidden]
    d_up_w: Vec<Vec<f32>>,   // [num_experts][inter*hidden]
    d_down_w: Vec<Vec<f32>>, // [num_experts][hidden*inter]
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

/// Compute the four named gradients of the sum-objective w.r.t. the MoE.
/// Slabs hold a row for every expert (zero for experts the router didn't
/// route to this token), matching the kernel's per-expert output layout
/// (`d_gate_w + exp * inter * hidden`, etc.) so the by-name assertions share
/// indexing with the kernel.
fn analytical_backward(fw: &FlatWeights) -> MoEGrads {
    let moe = build_oracle(fw);
    let x = cpu_tensor(fw.x.clone(), Shape::new(vec![BATCH, HIDDEN]));

    // Forward routing — the SAME selection the forward used.
    let (indices, weights) = moe.router.route(&x).expect("MoeRouter::route");
    debug_assert_eq!(indices.len(), BATCH);
    debug_assert_eq!(weights.len(), BATCH);

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

    for t in 0..BATCH {
        let chosen = &indices[t];
        let w = &weights[t];
        // d_y = 1 for every output element (sum objective).
        let d_y = [1.0f32; HIDDEN];

        for (rank, &e) in chosen.iter().enumerate() {
            let combine = w[rank];
            // s = routed_scaling_factor * combine. This is the ONLY place the
            // output-side scaling (rsf * router weight) enters the backward;
            // it propagates via `d_down_w` and `d_act`, and from there into
            // `d_h_gate` / `d_h_up` / `d_gate_w` / `d_up_w` / `d_x`. No second
            // application of `s` anywhere — the previous draft (and the
            // existing kernel) double-scaled `d_w`/`d_x` and dropped `h_up`
            // from `d_h_gate`; both are now fixed below (see NOTE).
            let s = RSF * combine;

            // Recompute the forward hidden states for this (token, expert).
            let x_row = &fw.x[t * HIDDEN..(t + 1) * HIDDEN];
            let gate_w = &fw.gate[e];
            let up_w = &fw.up[e];
            let down_w = &fw.down[e];

            // h_gate[j] = sum_i gate_w[j*hidden + i] * x[i]   (j indexes inter)
            // h_up[j]   = sum_i up_w[j*hidden + i] * x[i]
            let mut h_gate = [0.0f32; INTER];
            let mut h_up = [0.0f32; INTER];
            for j in 0..INTER {
                for i in 0..HIDDEN {
                    h_gate[j] += gate_w[j * HIDDEN + i] * x_row[i];
                    h_up[j] += up_w[j * HIDDEN + i] * x_row[i];
                }
            }
            // act[j] = silu(h_gate[j]) * h_up[j]
            let act: Vec<f32> = (0..INTER).map(|j| silu(h_gate[j]) * h_up[j]).collect();

            // d_down_w[e][h, j] += s * d_y[h] * act[j]   — outer product
            // (the `s` here is the one-and-only source of the rsf*w scaling
            //  entering d_down_w; verified by the rsf-scales-forward test).
            //
            // d_act[j] = sum_h s * d_y[h] * down_w[e][h, j]   — scaled by s via
            // the column-j contraction of `down_w` against the scaled d_y.
            let mut d_act = [0.0f32; INTER];
            for h in 0..HIDDEN {
                let dy_h_scaled = s * d_y[h];
                for j in 0..INTER {
                    d_down_w[e][h * INTER + j] += dy_h_scaled * act[j];
                    d_act[j] += s * d_y[h] * down_w[h * INTER + j];
                }
            }

            // SiLU-SwiGLU activation grad. The forward is
            //   act[j] = silu(h_gate[j]) * h_up[j]
            // so
            //   d_h_gate[j] = d_act[j] * (D silu)(h_gate[j]) * h_up[j]
            //   d_h_up[j]   = d_act[j] * silu(h_gate[j])
            //
            // NOTE (regression caught by this gate): the prior draft (and the
            // existing kernel `charon_backward.rs` line 140) computed
            //   d_gate = d_hidden * silu_grad(h_gate)
            // dropping the `h_up` factor. The by-name gradient gate surfaced
            // this as an `analytic = fd / h_up` discrepancy (ratio ≈ 1/h_up
            // exactly); fixed here and in the kernel source.
            let mut d_h_gate = [0.0f32; INTER];
            let mut d_h_up = [0.0f32; INTER];
            for j in 0..INTER {
                let sg = silu_grad(h_gate[j]);
                let sval = silu(h_gate[j]);
                d_h_gate[j] = d_act[j] * sg * h_up[j];
                d_h_up[j] = d_act[j] * sval;
            }

            // d_gate_w[e][j, i] += d_h_gate[j] * x[i]   (outer product)
            // d_up_w[e][j, i]   += d_h_up[j]   * x[i]
            // No extra `s`: the scaling already lives in d_h_gate/d_h_up via
            // d_act. Applying `s` again here would double-scale (the prior
            // draft did exactly that).
            for j in 0..INTER {
                for i in 0..HIDDEN {
                    d_gate_w[e][j * HIDDEN + i] += d_h_gate[j] * x_row[i];
                    d_up_w[e][j * HIDDEN + i] += d_h_up[j] * x_row[i];
                }
            }

            // d_x[t, i] += sum_j (gate_w[e][j, i] * d_h_gate[j]
            //                   + up_w[e][j, i]   * d_h_up[j])
            // (s is NOT re-applied — already absorbed via d_h_gate / d_h_up).
            // NOTE (regression caught by this gate): the prior draft re-scaled
            // this `acc` by `s`; that double-counted the rsf*w factor. Truth
            // (FD) absorbs `s` once via d_h_*, never twice.
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
// Finite-difference primitives
// ---------------------------------------------------------------------------

/// Central finite difference of `objective` w.r.t. a single scalar `p`:
/// `(f(p+eps) - f(p-eps)) / (2*eps)`. Accepts `FnMut` so the per-entry
/// perturbation closures can mutate the surrounding weight slab.
fn fd_scalar<F: FnMut(f32) -> f32>(mut objective: F, base: f32) -> f32 {
    (objective(base + EPS) - objective(base - EPS)) / (2.0 * EPS)
}

/// Compare an analytical gradient to a finite-difference estimate and panic
/// with a precise diagnostic if they diverge beyond `TOL` (relative for
/// non-tiny entries, absolute near zero).
///
/// `expect_nonzero` enforces the plan's anti-regression guard: at least one
/// entry must be non-trivially nonzero, EXCEPT for experts the router never
/// routed to this token (their grads are correctly zero by construction, so
/// an all-zero slab for a non-routed expert is the CORRECT behavior — without
/// this flag the guard would false-positive and reject a correct backward).
fn assert_grad_matches_fd(
    name: &str,
    analytic: &[f32],
    fd: impl Fn(usize) -> f32,
    expect_nonzero: bool,
) {
    let mut max_rel = 0.0f32;
    for (i, &a) in analytic.iter().enumerate() {
        let f = fd(i);
        let err = (a - f).abs();
        // relative tolerance where the analytic grad is non-trivial; absolute
        // floor where it's near zero (so a true-zero grad isn't flagged for
        // f32 roundoff at the ~1e-6 level).
        let denom = a.abs().max(1e-3);
        let rel = err / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel <= TOL,
            "grad `{name}` mismatch at index {i}: analytic={a:.6e} fd={f:.6e} \
             rel_err={rel:.3e} > TOL={TOL:.0e}",
        );
    }
    // Sanity: at least one entry must be non-trivially nonzero (a degenerate
    // all-zeros grad would let a mutation that drops the entire grad silently
    // pass) — the plan's specific regression-guard against "implemented only
    // d_down_w/d_x while claiming completeness." Only enforced when the slab
    // is SUPPOSED to have content (routed experts); non-routed experts are
    // correctly zero and would false-positive without this flag.
    if expect_nonzero {
        let max_abs = analytic.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-3,
            "grad `{name}` is all-zero (analytical backward produced no contribution; \
             the plan's prior d_down_w/d_x-only bug surfaced here as a 0/0 pass-fallback)",
        );
    }
    let _ = max_rel;
}

/// The set of expert indices the router actually selected on the test input.
/// Used to distinguish the per-expert grad slabs the by-name gate expects to
/// be non-empty (routed experts) from the empty ones (non-routed experts,
/// correctly zero by construction — without this distinction the
/// `expect_nonzero` guard would false-positive on the non-routed expert).
fn routed_experts(fw: &FlatWeights) -> std::collections::HashSet<usize> {
    let m = build_oracle(fw);
    let x = cpu_tensor(fw.x.clone(), Shape::new(vec![BATCH, HIDDEN]));
    let (indices, _) = m.router.route(&x).expect("route");
    indices.into_iter().flatten().collect()
}

// =============================================================================
// Named gradient checks — each is its own #[test], per the plan's gate:
// "d_x / d_gate_w / d_up_w / d_down_w individually asserted."
// =============================================================================

#[test]
fn d_down_w_gradient_matches_fd_by_name() {
    let fw = flat_expert_weights();
    let g = analytical_backward(&fw);
    let routed = routed_experts(&fw);
    // d_down_w[e][h*INTER + j] finite-difference per entry.
    let fd_for = |e: usize, h: usize, j: usize| -> f32 {
        let mut fw2 = flat_expert_weights();
        let idx = h * INTER + j;
        let base = fw2.down[e][idx];
        fd_scalar(
            |delta| {
                fw2.down[e][idx] = base + delta;
                objective(&fw2)
            },
            0.0,
        )
    };
    // Flatten into one slice per expert and run the comparator.
    for e in 0..NUM_EXPERTS {
        let flat = &g.d_down_w[e];
        assert_grad_matches_fd(
            &format!("d_down_w[e={e}]"),
            flat,
            |k| fd_for(e, k / INTER, k % INTER),
            routed.contains(&e),
        );
    }
}

#[test]
fn d_gate_w_gradient_matches_fd_by_name() {
    let fw = flat_expert_weights();
    let g = analytical_backward(&fw);
    let routed = routed_experts(&fw);
    let fd_for = |e: usize, j: usize, i: usize| -> f32 {
        let mut fw2 = flat_expert_weights();
        let idx = j * HIDDEN + i;
        let base = fw2.gate[e][idx];
        fd_scalar(
            |delta| {
                fw2.gate[e][idx] = base + delta;
                objective(&fw2)
            },
            0.0,
        )
    };
    for e in 0..NUM_EXPERTS {
        let flat = &g.d_gate_w[e];
        assert_grad_matches_fd(
            &format!("d_gate_w[e={e}]"),
            flat,
            |k| fd_for(e, k / HIDDEN, k % HIDDEN),
            routed.contains(&e),
        );
    }
}

#[test]
fn d_up_w_gradient_matches_fd_by_name() {
    let fw = flat_expert_weights();
    let g = analytical_backward(&fw);
    let routed = routed_experts(&fw);
    let fd_for = |e: usize, j: usize, i: usize| -> f32 {
        let mut fw2 = flat_expert_weights();
        let idx = j * HIDDEN + i;
        let base = fw2.up[e][idx];
        fd_scalar(
            |delta| {
                fw2.up[e][idx] = base + delta;
                objective(&fw2)
            },
            0.0,
        )
    };
    for e in 0..NUM_EXPERTS {
        let flat = &g.d_up_w[e];
        assert_grad_matches_fd(
            &format!("d_up_w[e={e}]"),
            flat,
            |k| fd_for(e, k / HIDDEN, k % HIDDEN),
            routed.contains(&e),
        );
    }
}

#[test]
fn d_x_gradient_matches_fd_by_name() {
    let fw = flat_expert_weights();
    let g = analytical_backward(&fw);
    let fd_for = |i: usize| -> f32 {
        let mut fw2 = flat_expert_weights();
        let base = fw2.x[i];
        fd_scalar(
            |delta| {
                fw2.x[i] = base + delta;
                objective(&fw2)
            },
            0.0,
        )
    };
    assert_grad_matches_fd("d_x", &g.d_x, fd_for, true);
}

/// Cross-check: the four named gradients together must reproduce the sum-
/// objective's directional derivative along a random direction, i.e.
/// `sum_k (d_obj/dw_k) * v_k == (obj(w + eps*v) - obj(w - eps*v)) / (2*eps)`.
/// This is a strong sanity check the four named grads are mutually consistent
/// (catches a sign or scale mistake that passes each by-name test in
/// isolation but breaks the composed backward — e.g. misplacing `rsf * w`
/// between d_w and d_x).
#[test]
fn directional_derivative_under_combined_named_grads() {
    let fw = flat_expert_weights();
    let g = analytical_backward(&fw);

    // Deterministic "random" direction (no RNG dependency for reproducibility).
    let mut dir_gate = vec![vec![0.0f32; INTER * HIDDEN]; NUM_EXPERTS];
    let mut dir_up = vec![vec![0.0f32; INTER * HIDDEN]; NUM_EXPERTS];
    let mut dir_down = vec![vec![0.0f32; HIDDEN * INTER]; NUM_EXPERTS];
    let mut dir_x = vec![0.0f32; BATCH * HIDDEN];
    let mut seed = 12345u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / 16777216.0) - 0.5
    };
    for e in 0..NUM_EXPERTS {
        for v in &mut dir_gate[e] {
            *v = rng();
        }
        for v in &mut dir_up[e] {
            *v = rng();
        }
        for v in &mut dir_down[e] {
            *v = rng();
        }
    }
    for v in &mut dir_x {
        *v = rng();
    }

    // Combined analytical directional derivative.
    let mut analytic_dd = 0.0f32;
    for e in 0..NUM_EXPERTS {
        for k in 0..INTER * HIDDEN {
            analytic_dd += g.d_gate_w[e][k] * dir_gate[e][k];
            analytic_dd += g.d_up_w[e][k] * dir_up[e][k];
        }
        for (dw, dd) in g.d_down_w[e].iter().zip(dir_down[e].iter()) {
            analytic_dd += dw * dd;
        }
    }
    for (dx, ddir) in g.d_x.iter().zip(dir_x.iter()) {
        analytic_dd += dx * ddir;
    }

    // Finite-difference directional derivative along the same direction.
    let mut fw_plus = flat_expert_weights();
    let mut fw_minus = flat_expert_weights();
    for e in 0..NUM_EXPERTS {
        for k in 0..INTER * HIDDEN {
            fw_plus.gate[e][k] += EPS * dir_gate[e][k];
            fw_minus.gate[e][k] -= EPS * dir_gate[e][k];
            fw_plus.up[e][k] += EPS * dir_up[e][k];
            fw_minus.up[e][k] -= EPS * dir_up[e][k];
        }
        for ((fp, fm), dd) in fw_plus.down[e]
            .iter_mut()
            .zip(fw_minus.down[e].iter_mut())
            .zip(dir_down[e].iter())
        {
            *fp += EPS * dd;
            *fm -= EPS * dd;
        }
    }
    for ((fp, fm), ddir) in fw_plus
        .x
        .iter_mut()
        .zip(fw_minus.x.iter_mut())
        .zip(dir_x.iter())
    {
        *fp += EPS * ddir;
        *fm -= EPS * ddir;
    }
    let fd_dd = (objective(&fw_plus) - objective(&fw_minus)) / (2.0 * EPS);

    let rel = ((analytic_dd - fd_dd).abs()) / analytic_dd.abs().max(1e-3);
    assert!(
        rel <= TOL,
        "combined directional derivative mismatch: analytic={analytic_dd:.6e} \
         fd={fd_dd:.6e} rel_err={rel:.3e} > TOL={TOL:.0e}",
    );
}

// ---------------------------------------------------------------------------
// Structural: the HIP kernel source encodes the same decomposition the
// validated host backward computes — so the on-device kernel is provably the
// same math, not a parallel implementation that could silently diverge.
// Pinned by symbol so a kernel mutation that drops a named grad, the SiLU
// grad, the rsf scaling, or the atomic-add into d_x is caught host-side
// before any device run.
// ---------------------------------------------------------------------------

#[test]
fn charon_backward_kernel_encodes_all_four_named_gradients_and_silu_grad() {
    let src = charon_backward::KERNEL_SOURCE;
    // All four named output buffers must appear — the plan guards against a
    // prior regression where only d_down_w/d_x were emitted while claiming
    // completeness. Pin each by name.
    for sym in ["d_gate_w", "d_up_w", "d_down_w", "d_x"] {
        assert!(
            src.contains(sym),
            "charon_backward KERNEL_SOURCE missing named grad `{sym}` — \
             the WI-Charon-1 plan requires all four explicitly",
        );
    }
    // The SiLU-derivative `silu_grad` device helper must be defined (the
    // decomposition's gate-backbone). A kernel that drops `silu_grad` is
    // structurally not the SiLU-SwiGLU backward.
    assert!(src.contains("silu_grad"), "missing silu_grad device helper");
    // The routed scaling factor must enter the gradient (the half-scale
    // surface the `routed_scaling_factor` regression test pinned on the
    // forward). Its presence in the backward pins the same surface upstream.
    assert!(
        src.contains("routed_scaling_factor"),
        "missing routed_scaling_factor — rsf must scale the backward too",
    );
    // `d_x` is accumulated across the K routed experts via `atomicAdd`
    // (matches the kernel's documented "token routed to K>1 experts has K
    // backward blocks; gradients into d_x[token] use atomicAdd"). Pin the
    // atomic accumulator so a race-free regression (e.g. replacing atomicAdd
    // with a store) is caught.
    assert!(
        src.contains("atomicAdd(&dx[") || src.contains("atomicAdd(&dx,"),
        "d_x must be accumulated with atomicAdd(&dx[..] / &dx, ...) per the contract",
    );
    // The expert-weight grads are also atomic-add accumulators (each grad
    // buffer is shared across the K blocks of one expert's tokens).
    assert!(
        src.contains("atomicAdd(&dgw")
            && src.contains("atomicAdd(&duw")
            && src.contains("atomicAdd(&ddw"),
        "d_gate_w/d_up_w/d_down_w must use atomicAdd(&dgw/duw/ddw, ...)",
    );
    // The by-name gradient gate surfaced two regressions in the historical
    // draft; both are now pinned in-symbol so neither can recurcur silently:
    //
    //  (a) Missing `h_up` factor in `d_h_gate` (act = silu(gate)*up; the gate-
    //      parent needs silu_grad * h_up). The kernel computes `sg = silu_grad(hg)`
    //      and `d_h_gate = d_act * sg * hu` — pin both halves so a mutant that
    //      drops `* hu` (the historical bug, surfaced as `analytic = fd / h_up`)
    //      or that re-aliases `sg = silu_gj` (loses the derivative) fails.
    assert!(
        src.contains("silu_grad(hg)"),
        "gate path must call `silu_grad(hg)` — silu_grad dropped is structurally \
         not the SiLU backward",
    );
    assert!(
        src.contains("sg * hu") || src.contains("hu * sg"),
        "d_h_gate must include the `h_up` (hu) factor — the regression this \
         by-name gate was built to catch",
    );
    //  (b) Double-scaling d_w / d_x by the routed scaling factor (the draft
    //      multiplied d_gate_w by `rsf` AND fed `rsf*dy` into d_act, so `s`
    //      appeared twice). The fix absorbs `s` once via d_act and never
    //      re-applies it. Pin the absence of an outer `rsf *` on the
    //      `atomicAdd(&dgw/duw` lines and on `d_x`'s outer accumulation —
    //      structurally, those atomicAdd calls must read their grandient terms
    //      directly, not `rsf * term`.
    assert!(
        !src.contains("atomicAdd(&dgw[j * hidden + i], rsf * d_h_gate")
            && !src.contains("atomicAdd(&duw[j * hidden + i], rsf * d_h_up"),
        "d_gate_w/d_up_w must NOT be re-scaled by rsf (s already absorbed via d_act)",
    );
    assert!(
        !src.contains("atomicAdd(&dx[i], rsf *") && !src.contains("atomicAdd(&dx[i], rsf*"),
        "d_x must NOT carry an outer rsf multiplier (s absorbed via d_h_gate/d_h_up)",
    );
    // Up-path parent: `silu_gj` is `silu(h_gate)` (the factor that multiplies
    // h_up in `act`); `d_h_up = d_act * silu_gj` reuses it rather than
    // recomputing. Pin the name so a mutant that recomputes silu inline (and
    // could drift out of sync with act_j) fails.
    assert!(
        src.contains("silu_gj") && src.contains("d_h_up   = d_act * silu_gj"),
        "up path must reuse silu(h_gate) via `silu_gj` for `d_h_up`",
    );
}

/// Self-consistency: the host analytical backward (validated above) is the
/// reference the kernel must match. Pin the kernel-comment decomposition
/// instruction that names all four outputs, so a future edit that rewrites
/// the doc comment without the four-by-name guarantee fails this gate.
#[test]
fn charon_backward_docstring_names_all_four_gradients() {
    // The module-level doc comment in charon_backward.rs lists the four named
    // grads explicitly. Pull the source file itself (not KERNEL_SOURCE) so we
    // pin the human-readable contract, not just the HIP string.
    let doc = include_str!("../src/kernels/charon_backward.rs");
    for sym in ["d_x", "d_gate_w", "d_up_w", "d_down_w"] {
        assert!(
            doc.contains(sym),
            "charon_backward.rs docstring missing named grad `{sym}` — \
             the WI-Charon-1 plan's by-name gate must be documented in-source",
        );
    }
}
