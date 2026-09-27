//! Batched Gated DeltaNet (KDA) decode step: ROCm kernels vs an f64 reference.
//!
//! This is the op that lets a Qwen3.5/3.8 recurrent layer decode without
//! reading its state back to the host. Two things about it are new relative to
//! `grim_kda_gated_delta_rule_step`, and both are what the test exists for:
//!
//! * it folds in everything the per-head op left to the caller — the per-head
//!   L2 norms of q and k, `softplus(alpha + dt_bias) * ssm_a`, the sigmoid on
//!   beta, the output RMS norm with its `ssm_norm` weight, and the `silu(z)`
//!   gate. A wrong sign or a missing `ssm_a` here is invisible in the output
//!   values of a zero-state single step, so the state is seeded NON-ZERO and
//!   run for several steps.
//! * the recurrent state is written in place. An implementation that computed
//!   the right output from a scratch copy of the state would pass step 1 and
//!   diverge from step 2 on, so the chain is asserted step by step.
//!
//! The reference is written here in f64 from the published update (ICLR 2025,
//! Eq. 10) rather than by calling the CPU backend, so this is a true oracle
//! and not "CPU == ROCm" with a shared mistake.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::{RecurrentOps, RocmDevice};
use grim_tensor::CoreTensorOps;
use grim_tensor::{DType, Shape};
use std::sync::Arc;

const NV: usize = 6;
const NK: usize = 2;
const D: usize = 16;
const EPS: f32 = 1e-6;
const STEPS: usize = 4;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

fn conv_dim() -> usize {
    2 * NK * D + NV * D
}

fn linspace(n: usize, lo: f64, hi: f64) -> Vec<f32> {
    if n == 1 {
        return vec![lo as f32];
    }
    (0..n)
        .map(|i| (lo + (hi - lo) * (i as f64) / ((n - 1) as f64)) as f32)
        .collect()
}

/// Independent f64 reference for the whole batched step, including state.
fn reference_step(
    conv: &[f32],
    alpha: &[f32],
    beta: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    norm_w: &[f32],
    z: Option<&[f32]>,
    state: &mut [f64],
) -> Vec<f64> {
    // The op applies the SiLU as it reads the stream (the reference's
    // `ggml_silu` before the q/k/v split), so the reference does too.
    let conv: Vec<f64> = conv
        .iter()
        .map(|&x| {
            let x = x as f64;
            x / (1.0 + (-x).exp())
        })
        .collect();
    let conv = &conv[..];
    let key_dim = NK * D;
    let eps = EPS as f64;
    let softplus = |x: f64| {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            x.exp().ln_1p()
        }
    };
    let l2 = |v: &[f64]| {
        let ss: f64 = v.iter().map(|x| x * x).sum();
        let den = (ss + eps).sqrt();
        if den <= 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / den).collect()
        }
    };

    let mut out = vec![0.0f64; NV * D];
    for h in 0..NV {
        let k_base = (h % NK) * D;
        let q_base = 2 * key_dim + h * D;
        let k_raw: Vec<f64> = conv[k_base..k_base + D].iter().map(|&v| v as f64).collect();
        let qv: Vec<f64> = conv[q_base..q_base + D].iter().map(|&v| v as f64).collect();

        let gate = softplus(alpha[h] as f64 + dt_bias[h] as f64) * ssm_a[h] as f64;
        let beta_val = 1.0 / (1.0 + (-(beta[h] as f64)).exp());
        let decay = gate.exp();
        let k_l2 = l2(&k_raw);
        let q_l2 = l2(&qv);

        let head = &mut state[h * D * D..(h + 1) * D * D];
        let mut acc = vec![0.0f64; D];
        for j in 0..D {
            let row = &mut head[j * D..(j + 1) * D];
            let pred: f64 = k_l2
                .iter()
                .zip(row.iter())
                .map(|(k, s)| k * (decay * s))
                .sum();
            let delta = beta_val * (qv[j] - pred);
            for i in 0..D {
                row[i] = decay * row[i] + k_l2[i] * delta;
                acc[j] += q_l2[i] * row[i];
            }
            // Reference: the head output is scaled by 1/sqrt(S_v) before the
            // gated norm (llama.cpp `gated_delta_net.cu:281`).
            acc[j] *= 1.0 / (D as f64).sqrt();
        }
        let ss: f64 = acc.iter().map(|a| a * a).sum();
        let inv = 1.0 / (ss / D as f64 + eps).sqrt();
        for i in 0..D {
            let g = match z {
                Some(zz) => {
                    let zv = zz[h * D + i] as f64;
                    zv / (1.0 + (-zv).exp())
                }
                None => 1.0,
            };
            out[h * D + i] = acc[i] * inv * norm_w[i] as f64 * g;
        }
    }
    out
}

struct Case {
    conv: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    norm_w: Vec<f32>,
    z: Vec<f32>,
}

fn case(step: usize) -> Case {
    // Distinct stream per step so a state that silently resets is caught.
    let s = step as f64;
    Case {
        conv: linspace(conv_dim(), -1.5 + 0.3 * s, 2.0 - 0.4 * s),
        alpha: linspace(NV, -0.4 + 0.1 * s, 0.6),
        beta: linspace(NV, -1.0, 1.0),
        dt_bias: linspace(NV, 0.1, 0.9),
        ssm_a: linspace(NV, 0.5, 1.5),
        norm_w: linspace(NV * D, 0.8, 1.2),
        z: linspace(NV * D, -0.7, 0.7),
    }
}

fn upload(
    dev: &Arc<RocmDevice>,
    data: &[f32],
    shape: Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev.as_ref(), data, &shape, DType::F32).expect("from_cpu")
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn rocm_kda_batched_matches_f64_over_a_state_chain() {
    let Some(dev) = gpu_device() else { return };
    eprintln!("[kda-batched] device ok");

    // Non-zero seed: a zero state makes the decayed key dot and the carried
    // state indistinguishable, so a stale-state variant would survive step 1.
    let seed: Vec<f32> = linspace(NV * D * D, -0.3, 0.3);
    let state = upload(&dev, &seed, Shape::new(vec![NV, D, D]));
    eprintln!("[kda-batched] state uploaded ({} floats)", seed.len());

    let mut want_state: Vec<f64> = seed.iter().map(|&v| v as f64).collect();

    for step in 0..STEPS {
        let c = case(step);
        let conv = upload(&dev, &c.conv, Shape::new(vec![conv_dim()]));
        let alpha = upload(&dev, &c.alpha, Shape::new(vec![NV]));
        let beta = upload(&dev, &c.beta, Shape::new(vec![NV]));
        let dt_bias = upload(&dev, &c.dt_bias, Shape::new(vec![NV]));
        let ssm_a = upload(&dev, &c.ssm_a, Shape::new(vec![NV]));
        let norm_w = upload(&dev, &c.norm_w, Shape::new(vec![NV * D]));
        let z = upload(&dev, &c.z, Shape::new(vec![NV * D]));

        let want = reference_step(
            &c.conv,
            &c.alpha,
            &c.beta,
            &c.dt_bias,
            &c.ssm_a,
            &c.norm_w,
            Some(&c.z),
            &mut want_state,
        );
        eprintln!("[kda-batched] step {step}: launching");
        let (out, _h) = dev
            .kda_gated_delta_rule_batched(
                conv.as_ref(),
                alpha.as_ref(),
                beta.as_ref(),
                dt_bias.as_ref(),
                ssm_a.as_ref(),
                norm_w.as_ref(),
                Some(z.as_ref()),
                state.as_ref(),
                NV,
                NK,
                D,
                EPS,
                &Shape::new(vec![NV * D]),
            )
            .expect("kda_gated_delta_rule_batched");
        eprintln!("[kda-batched] step {step}: launched, reading back");
        let got = out.to_cpu_vec_f32().expect("read output");
        eprintln!("[kda-batched] step {step}: read back");

        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (g as f64 - w).abs();
            let rel = diff / w.abs().max(1e-3);
            let score = diff.min(rel);
            if score > worst {
                worst = score;
                worst_at = i;
            }
        }
        assert!(
            worst <= 1e-4,
            "step {step}: output worst |err| {worst:.3e} at {worst_at} \
             (got {}, want {}); a chained-state or gate error shows up here",
            got[worst_at],
            want[worst_at]
        );
    }

    // The state must have been carried and written, not recomputed from zero.
    let got_state = state.to_cpu_vec_f32().expect("read state");
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got_state.iter().zip(want_state.iter()).enumerate() {
        let diff = (g as f64 - w).abs();
        let rel = diff / w.abs().max(1e-3);
        let score = diff.min(rel);
        if score > worst {
            worst = score;
            worst_at = i;
        }
    }
    assert!(
        worst <= 1e-4,
        "recurrent state worst |err| {worst:.3e} at {worst_at} \
         (got {}, want {}); the in-place state update is wrong",
        got_state[worst_at],
        want_state[worst_at]
    );
    eprintln!("[kda-batched] {STEPS} chained steps + state match f64 (worst {worst:.2e})");
}

/// `z = None` must mean a gate of 1.0, not `silu(0) = 0` — the latter would
/// silently zero the whole recurrent branch.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn rocm_kda_batched_without_z_keeps_the_branch_alive() {
    let Some(dev) = gpu_device() else { return };
    let c = case(0);
    let seed: Vec<f32> = linspace(NV * D * D, -0.3, 0.3);

    let state_a = upload(&dev, &seed, Shape::new(vec![NV, D, D]));
    let state_b = upload(&dev, &seed, Shape::new(vec![NV, D, D]));
    let conv = upload(&dev, &c.conv, Shape::new(vec![conv_dim()]));
    let alpha = upload(&dev, &c.alpha, Shape::new(vec![NV]));
    let beta = upload(&dev, &c.beta, Shape::new(vec![NV]));
    let dt_bias = upload(&dev, &c.dt_bias, Shape::new(vec![NV]));
    let ssm_a = upload(&dev, &c.ssm_a, Shape::new(vec![NV]));
    let norm_w = upload(&dev, &c.norm_w, Shape::new(vec![NV * D]));
    let z = upload(&dev, &c.z, Shape::new(vec![NV * D]));

    let call = |st: &dyn grim_tensor::BackendStorage,
                with_z: Option<&dyn grim_tensor::BackendStorage>| {
        dev.kda_gated_delta_rule_batched(
            conv.as_ref(),
            alpha.as_ref(),
            beta.as_ref(),
            dt_bias.as_ref(),
            ssm_a.as_ref(),
            norm_w.as_ref(),
            with_z,
            st,
            NV,
            NK,
            D,
            EPS,
            &Shape::new(vec![NV * D]),
        )
        .expect("kda batched")
        .0
        .to_cpu_vec_f32()
        .expect("read")
    };
    let with_z = call(state_a.as_ref(), Some(z.as_ref()));
    let without_z = call(state_b.as_ref(), None);

    // z gates the output only, so the two states must be bit-identical.
    let sa = state_a.to_cpu_vec_f32().expect("state a");
    let sb = state_b.to_cpu_vec_f32().expect("state b");
    assert_eq!(sa, sb, "z must not feed the recurrent state");

    assert!(
        with_z.iter().all(|v| v.abs() > 1e-6),
        "z-gated output is all zero; the branch is dead"
    );
    let differs = with_z
        .iter()
        .zip(without_z.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(differs, "z had no effect on the output");
}
