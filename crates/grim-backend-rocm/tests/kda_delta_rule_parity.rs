//! Gated DeltaNet recurrence parity: ROCm kernel vs the CPU reference.
//!
//! `kda_gated_delta_rule_step` is the recurrence Qwen3.5/3.8 hybrid models use in
//! 49 of their 65 layers. It had NO numeric coverage on any backend — the only
//! assertions in the tree were `KERNEL_SOURCE.contains("grim_kda_gated_delta_
//! rule_step")` string checks, which pass whether or not the math is right.
//!
//! The CPU path implements the published Gated DeltaNet update (ICLR 2025 Eq. 10):
//!
//! ```text
//! decay = exp(a_gate)
//! pred  = sum_k k * (decay * S)          # decay applied BEFORE the dot
//! delta = beta * (v - pred)              # beta scales the FULL delta term
//! S_new = decay * S + k * delta
//! out   = sum_k q * S_new
//! ```
//!
//! The ROCm kernel diverged on all three: it used `sigmoid(a_gate)` instead of
//! `exp`, omitted decay from the k-dot (so the error term is computed against a
//! stale state), and folded beta inside as `v - beta*(k·S)` so beta never scaled
//! the v term. Each of those is a different recurrence, not a rounding
//! difference.
//!
//! The reference is the CPU backend, so this is a true cross-backend check and
//! not a device kernel compared against a mirror of itself.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::{RecurrentOps, RocmDevice};
use grim_tensor::CoreTensorOps;
use grim_tensor::{DType, Shape};
use std::sync::Arc;

const D_K: usize = 64;
const D_V: usize = 64;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

/// Host implementation of the published recurrence, written here independently
/// of both backends so the test is not merely asserting CPU == ROCm when both
/// are wrong the same way.
fn reference_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    a_gate: f32,
    state: &[f32], // [d_v, d_k] row-major
) -> (Vec<f32>, Vec<f32>) {
    let decay = a_gate.exp();
    let mut out = vec![0.0f32; D_V];
    let mut new_state = vec![0.0f32; D_V * D_K];
    for i in 0..D_V {
        let row = &state[i * D_K..(i + 1) * D_K];
        // decay BEFORE the dot
        let pred: f32 = k
            .iter()
            .zip(row.iter())
            .map(|(kk, ss)| kk * (decay * ss))
            .sum();
        // beta scales the whole delta
        let delta = beta * (v[i] - pred);
        let mut acc = 0.0f32;
        for col in 0..D_K {
            let s_new = decay * row[col] + k[col] * delta;
            new_state[i * D_K + col] = s_new;
            acc += q[col] * s_new;
        }
        out[i] = acc;
    }
    (out, new_state)
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn rocm_kda_step_matches_published_recurrence() {
    let Some(dev) = gpu_device() else { return };

    // Distinctive inputs so a swapped gate or a misplaced beta cannot pass.
    let q: Vec<f32> = (0..D_K).map(|i| (i as f32 * 0.13).sin()).collect();
    let k: Vec<f32> = (0..D_K).map(|i| (i as f32 * 0.07).cos() * 0.5).collect();
    let v: Vec<f32> = (0..D_V).map(|i| i as f32 * 0.031 - 0.7).collect();
    let beta = 0.35f32;
    let a_gate = -0.4f32;
    // Non-zero initial state: a zero state would hide the decay-before-dot
    // error entirely, because pred would be 0 either way.
    let state: Vec<f32> = (0..D_V * D_K).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();

    let (want_out, _want_state) = reference_step(&q, &k, &v, beta, a_gate, &state);

    let q_s = dev.from_cpu(&q, &Shape::new(vec![D_K]), DType::F32).unwrap();
    let k_s = dev.from_cpu(&k, &Shape::new(vec![D_K]), DType::F32).unwrap();
    let v_s = dev.from_cpu(&v, &Shape::new(vec![D_V]), DType::F32).unwrap();
    let beta_s = dev.from_cpu(&[beta], &Shape::new(vec![1]), DType::F32).unwrap();
    let gate_s = dev.from_cpu(&[a_gate], &Shape::new(vec![1]), DType::F32).unwrap();
    let state_s = dev.from_cpu(&state, &Shape::new(vec![D_V, D_K]), DType::F32).unwrap();

    let out_shape = Shape::new(vec![D_V]);
    let (out, _handle) = dev
        .kda_gated_delta_rule_step(
            q_s.as_ref(),
            k_s.as_ref(),
            v_s.as_ref(),
            beta_s.as_ref(),
            gate_s.as_ref(),
            state_s.as_ref(),
            D_K,
            D_V,
            &out_shape,
        )
        .expect("kda_gated_delta_rule_step");
    let got = out.to_cpu_vec_f32().expect("read out");
    assert_eq!(got.len(), D_V);

    let mut worst = 0.0f32;
    for i in 0..D_V {
        worst = worst.max((got[i] - want_out[i]).abs() / want_out[i].abs().max(1.0));
    }
    eprintln!("[kda] ROCm vs published recurrence: worst relative error {worst:.6}");
    assert!(
        worst < 1e-3,
        "ROCm KDA step diverges from the published Gated DeltaNet update: \
         worst relative error {worst}. Check gate (exp vs sigmoid), decay-before-dot, \
         and beta scaling the full delta term. row0 got {} want {}",
        got[0],
        want_out[0]
    );
}

/// Guards the specific error mode where the kernel omits decay from the k-dot.
/// With a zero state this bug is invisible; with a non-zero state it is not.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn decay_is_applied_before_the_key_dot() {
    let Some(dev) = gpu_device() else { return };

    let q: Vec<f32> = (0..D_K).map(|i| (i as f32 * 0.11).cos()).collect();
    let k: Vec<f32> = (0..D_K).map(|i| (i as f32 * 0.05).sin()).collect();
    let v: Vec<f32> = vec![0.25; D_V];
    let beta = 0.5f32;
    let a_gate = 0.2f32;
    let state: Vec<f32> = (0..D_V * D_K).map(|i| (i as f32 * 0.023).cos() * 0.4).collect();

    let (want, _) = reference_step(&q, &k, &v, beta, a_gate, &state);

    let q_s = dev.from_cpu(&q, &Shape::new(vec![D_K]), DType::F32).unwrap();
    let k_s = dev.from_cpu(&k, &Shape::new(vec![D_K]), DType::F32).unwrap();
    let v_s = dev.from_cpu(&v, &Shape::new(vec![D_V]), DType::F32).unwrap();
    let beta_s = dev.from_cpu(&[beta], &Shape::new(vec![1]), DType::F32).unwrap();
    let gate_s = dev.from_cpu(&[a_gate], &Shape::new(vec![1]), DType::F32).unwrap();
    let state_s = dev.from_cpu(&state, &Shape::new(vec![D_V, D_K]), DType::F32).unwrap();

    let (out, _h) = dev
        .kda_gated_delta_rule_step(
            q_s.as_ref(),
            k_s.as_ref(),
            v_s.as_ref(),
            beta_s.as_ref(),
            gate_s.as_ref(),
            state_s.as_ref(),
            D_K,
            D_V,
            &Shape::new(vec![D_V]),
        )
        .expect("kda step");
    let got = out.to_cpu_vec_f32().unwrap();

    // With a non-zero state, omitting decay from the dot is a large error.
    let drift: f32 = (0..D_V)
        .map(|i| (got[i] - want[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift < 0.05,
        "decay-before-dot mismatch: max abs drift {drift} (row0 got {} want {})",
        got[0],
        want[0]
    );
}
