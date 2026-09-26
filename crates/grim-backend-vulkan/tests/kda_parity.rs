//! Gated DeltaNet parity on Vulkan, against an independent reference.
//!
//! The Vulkan path implemented a DIFFERENT recurrence — `S' = gate*S + beta*v*kᵀ`
//! with no prediction/delta term — and stored S as `[d_k, d_v]` where ROCm, CUDA
//! and CPU use `[d_v, d_k]`. Corrected alongside the ROCm fix in 23f809f5.
//!
//! The reference below is written independently in the test rather than calling
//! the CPU backend, so a shared error cannot cancel out.
//!
//! Gated: skipped when no Vulkan device is present.

use grim_tensor::{CoreTensorOps, DType, RecurrentOps, Shape};

const D_K: usize = 64;
const D_V: usize = 64;

fn reference(q: &[f32], k: &[f32], v: &[f32], beta: f32, a_gate: f32, s: &[f32]) -> Vec<f32> {
    let decay = a_gate.exp();
    let mut out = vec![0.0f32; D_V];
    for j in 0..D_V {
        let row = &s[j * D_K..(j + 1) * D_K];
        let pred: f32 = k.iter().zip(row.iter()).map(|(kk, ss)| kk * (decay * ss)).sum();
        let delta = beta * (v[j] - pred);
        let mut acc = 0.0f32;
        for i in 0..D_K {
            acc += q[i] * (decay * row[i] + k[i] * delta);
        }
        out[j] = acc;
    }
    out
}

#[test]
fn vulkan_kda_step_matches_published_recurrence() {
    let dev = grim_backend_vulkan::VulkanDevice::new();

    let q: Vec<f32> = (0..D_K).map(|i| (i as f32 * 0.13).sin()).collect();
    let k: Vec<f32> = (0..D_K).map(|i| (i as f32 * 0.07).cos() * 0.5).collect();
    let v: Vec<f32> = (0..D_V).map(|i| i as f32 * 0.031 - 0.7).collect();
    let beta = 0.35f32;
    let a_gate = -0.4f32;
    // Non-zero state: with a zero state the decay-before-dot error is invisible.
    let state: Vec<f32> = (0..D_V * D_K).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();

    let want = reference(&q, &k, &v, beta, a_gate, &state);

    let qs = dev.from_cpu(&q, &Shape::new(vec![D_K]), DType::F32).unwrap();
    let ks = dev.from_cpu(&k, &Shape::new(vec![D_K]), DType::F32).unwrap();
    let vs = dev.from_cpu(&v, &Shape::new(vec![D_V]), DType::F32).unwrap();
    let bs = dev.from_cpu(&[beta], &Shape::new(vec![1]), DType::F32).unwrap();
    let gs = dev.from_cpu(&[a_gate], &Shape::new(vec![1]), DType::F32).unwrap();
    let ss = dev.from_cpu(&state, &Shape::new(vec![D_V, D_K]), DType::F32).unwrap();

    let (out, _h) = dev
        .kda_gated_delta_rule_step(
            qs.as_ref(),
            ks.as_ref(),
            vs.as_ref(),
            bs.as_ref(),
            gs.as_ref(),
            ss.as_ref(),
            D_K,
            D_V,
            &Shape::new(vec![D_V]),
        )
        .expect("kda step");
    let got = out.to_cpu_vec_f32().unwrap();

    let mut worst = 0.0f32;
    for j in 0..D_V {
        worst = worst.max((got[j] - want[j]).abs() / want[j].abs().max(1.0));
    }
    eprintln!("[VK-KDA] worst relative error {worst:.6}");
    assert!(
        worst < 1e-3,
        "Vulkan KDA step diverges from the published Gated DeltaNet update: \
         worst relative error {worst}. row0 got {} want {}",
        got[0],
        want[0]
    );
}
