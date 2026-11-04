//! Task 2.7 Parity Gate: KDA decode (short conv + `kda_gated_delta_rule_batched_into`)
//! vs an independent f64 CPU oracle.
//!
//! What was here before: device init + synchronize + eprintln, asserting nothing.
//!
//! What this is now: the Standard trio as named in the header —
//! `short_conv1d_causal_step_into` (raw conv sum, SiLU applied by the caller
//! side / the KDA op itself) feeding `kda_gated_delta_rule_batched_into` — run
//! on a small shape against a CPU f64 reference written inline from the
//! published update (ICLR 2025, Eq. 10 style, as in `kda_batched_parity`),
//! with NON-ZERO seeded recurrent state and a chained second step so a
//! stale-state or ring-shift mistake diverges and is caught.
//!
//! NOT reachable (factual locus, not a skip): `launch_gla_persistent_megakernel`
//! (`src/device/compute/gla_mega_launchers.rs`) exposes only the GRAVE Phase-5
//! GLA instruction set (OP_RMS_NORM, OP_DOT4_GEMV, OP_GDN2_RECURRENT,
//! OP_SHORTCONV_FUSED, OP_FFN_GATEUP_SILU, OP_RESIDUAL_ADD — see
//! `src/kernels/gla_mega_kernel.rs`). There is no KDA / gated-delta-rule opcode
//! in the megakernel, so a megakernel-vs-trio KDA comparison cannot be run
//! against any exported launcher today. The upgrade when a KDA opcode lands in
//! `GLA_MEGA_KERNEL_SOURCE` is to add a third arm: same inputs into the mega
//! path, compare its output against this same oracle.
//!
//! Tolerance: worst relative-ish score ≤ 1e-4 (f32 kernel vs f64 oracle), the
//! same bound as `kda_batched_parity`. The trio's wall latency is printed for
//! the record (per res7 §10a it is the fine-grained path that wins on
//! memory-bound decode); no latency assertion is made here.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device, `#[ignore]` so it never runs
//! in a CPU/offline CI lane. Local manual runs only (consumer ROCm project).

use grim_backend_rocm::{RocmDevice, RocmStorage, as_rocm};
use grim_tensor::CoreTensorOps;
use grim_tensor::{BackendStorage, DType, Shape};
use std::sync::Arc;

const NV: usize = 6;
const NK: usize = 2;
const D: usize = 16;
const KS: usize = 4;
const CHANNELS: usize = 2 * NK * D + NV * D;
const EPS: f32 = 1e-6;
const STEPS: usize = 2;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device().map(Arc::new)
}

fn rocm_of(s: &Box<dyn BackendStorage>) -> &RocmStorage {
    as_rocm(s.as_ref()).expect("RocmStorage")
}

fn linspace(n: usize, lo: f64, hi: f64) -> Vec<f32> {
    if n == 1 {
        return vec![lo as f32];
    }
    (0..n)
        .map(|i| (lo + (hi - lo) * (i as f64) / ((n - 1) as f64)) as f32)
        .collect()
}

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Independent f64 reference for the whole batched KDA step, including state.
/// Written here from the math (ICLR 2025, Eq. 10) rather than by calling the
/// CPU backend, so agreement is evidence. Copied in spirit from
/// `kda_batched_parity.rs` which is the exemplar.
fn reference_step(
    conv: &[f32],
    alpha: &[f32],
    beta: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    norm_w: &[f32],
    z: &[f32],
    state: &mut [f64],
) -> Vec<f64> {
    // The op applies the SiLU as it reads the stream (the reference's
    // `ggml_silu` before the q/k/v split), so the reference does too.
    let conv: Vec<f64> = conv
        .iter()
        .map(|&x| silu64(x as f64))
        .collect();
    let conv: &[f64] = &conv;
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
        // [q | k | v]: q at offset 0, k at key_dim, v at 2*key_dim.
        let q_base = (h % NK) * D;
        let k_base = key_dim + (h % NK) * D;
        let v_base = 2 * key_dim + h * D;
        let q_raw: Vec<f64> = conv[q_base..q_base + D].to_vec();
        let k_raw: Vec<f64> = conv[k_base..k_base + D].to_vec();
        let qv: Vec<f64> = conv[v_base..v_base + D].to_vec();

        let gate = softplus(alpha[h] as f64 + dt_bias[h] as f64) * ssm_a[h] as f64;
        let beta_val = 1.0 / (1.0 + (-(beta[h] as f64)).exp());
        let decay = gate.exp();
        let k_l2 = l2(&k_raw);
        let q_l2 = l2(&q_raw);

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
            // The head output is scaled by 1/sqrt(S_v) before the gated norm.
            acc[j] *= 1.0 / (D as f64).sqrt();
        }
        let ss: f64 = acc.iter().map(|a| a * a).sum();
        let inv = 1.0 / (ss / D as f64 + eps).sqrt();
        for i in 0..D {
            let zv = z[h * D + i] as f64;
            out[h * D + i] = acc[i] * inv * norm_w[i] as f64 * silu64(zv);
        }
    }
    out
}

/// CPU oracle for one raw short-conv step, matching the device kernel's
/// documented contract (see `kda_conv_input_diff.rs`): the kernel outputs the
/// pre-SiLU sum and updates the ring in place. Weights are [channels, ks]
/// row-major; the ring is [channels, ks-1] with slot 0 = oldest.
fn conv_oracle_step(
    x: &[f32],
    w: &[f32],
    ring: &mut [f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; CHANNELS];
    for c in 0..CHANNELS {
        let mut sum = x[c] * w[c * KS + (KS - 1)];
        for k in 0..KS - 1 {
            sum += ring[c * (KS - 1) + k] * w[c * KS + k];
        }
        out[c] = sum;
    }
    // In-place ring shift: post[c][0..ks-2] = pre[c][1..ks-1], post[last] = x.
    for c in 0..CHANNELS {
        for k in 0..KS - 2 {
            ring[c * (KS - 1) + k] = ring[c * (KS - 1) + k + 1];
        }
        ring[c * (KS - 1) + (KS - 2)] = x[c];
    }
    out
}

struct Case {
    conv_in: Vec<f32>,
    conv_w: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    norm_w: Vec<f32>,
    z: Vec<f32>,
}

fn case(step: usize) -> Case {
    // Distinct streams per step so a state/ring that silently resets is caught.
    let s = step as f64;
    Case {
        conv_in: linspace(CHANNELS, -0.9 + 0.3 * s, 0.9 - 0.2 * s),
        conv_w: linspace(CHANNELS * KS, -0.4 + 0.1 * s, 0.4),
        alpha: linspace(NV, -0.4 + 0.1 * s, 0.6),
        beta: linspace(NV, -1.0, 1.0),
        dt_bias: linspace(NV, 0.1, 0.9),
        ssm_a: linspace(NV, 0.5, 1.5),
        // Per-HEAD width (D), not per-value-stream (NV*D).
        norm_w: linspace(D, 0.8, 1.2),
        z: linspace(NV * D, -0.7, 0.7),
    }
}

fn upload(
    dev: &Arc<RocmDevice>,
    data: &[f32],
) -> Box<dyn BackendStorage> {
    dev.from_cpu(data, &Shape::new(vec![data.len()]), DType::F32)
        .expect("from_cpu")
}

fn from_cpu_bytes_f32(
    dev: &Arc<RocmDevice>,
    data: &[f32],
) -> Box<dyn BackendStorage> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    grim_backend_rocm::MemoryOps::from_cpu_bytes(
        dev.as_ref(),
        &bytes,
        &Shape::new(vec![data.len()]),
        DType::F32,
    )
    .expect("from_cpu_bytes")
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 (local manual only; no new CI)"]
fn kda_trio_matches_f64_oracle_over_conv_and_state_chain() {
    let Some(dev) = gpu_device() else { return };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    eprintln!("[kda-mega-trio] device ok");

    // Non-zero seeds: a zero state/ring makes decay and shift
    // indistinguishable, so a stale-state variant would survive step 1.
    let conv_ring0: Vec<f32> = linspace(CHANNELS * (KS - 1), -0.2, 0.2);
    let state0: Vec<f32> = linspace(NV * D * D, -0.3, 0.3);

    let conv_ring = from_cpu_bytes_f32(&dev, &conv_ring0);
    let conv_out = from_cpu_bytes_f32(&dev, &vec![0.0f32; CHANNELS]);
    let state = from_cpu_bytes_f32(&dev, &state0);
    let acc_scratch = from_cpu_bytes_f32(&dev, &vec![0.0f32; NV * D]);
    let out = from_cpu_bytes_f32(&dev, &vec![0.0f32; NV * D]);

    // Oracle-side mirrors, updated step by step.
    let mut want_ring: Vec<f32> = conv_ring0.clone();
    let mut want_state: Vec<f64> = state0.iter().map(|&v| v as f64).collect();

    let mut trio_us: f64 = 0.0;
    for step in 0..STEPS {
        let c = case(step);
        let conv_in_dev = from_cpu_bytes_f32(&dev, &c.conv_in);
        let conv_w_dev = from_cpu_bytes_f32(&dev, &c.conv_w);
        let alpha = upload(&dev, &c.alpha);
        let beta = upload(&dev, &c.beta);
        let dt_bias = upload(&dev, &c.dt_bias);
        let ssm_a = upload(&dev, &c.ssm_a);
        let norm_w = upload(&dev, &c.norm_w);
        let z = upload(&dev, &c.z);

        // Oracle: raw conv step (ring shifts here), then the f64 KDA step.
        let conv_raw = conv_oracle_step(&c.conv_in, &c.conv_w, &mut want_ring);
        let want = reference_step(
            &conv_raw,
            &c.alpha,
            &c.beta,
            &c.dt_bias,
            &c.ssm_a,
            &c.norm_w,
            &c.z,
            &mut want_state,
        );

        eprintln!("[kda-mega-trio] step {step}: launching conv + batched trio");
        let t0 = std::time::Instant::now();
        dev.short_conv1d_causal_step_into(
            conv_in_dev.as_ref(),
            conv_w_dev.as_ref(),
            None,
            conv_ring.as_ref(),
            rocm_of(&conv_out),
        )
        .expect("short_conv1d_causal_step_into");
        dev.kda_gated_delta_rule_batched_into(
            conv_out.as_ref(),
            alpha.as_ref(),
            beta.as_ref(),
            dt_bias.as_ref(),
            ssm_a.as_ref(),
            norm_w.as_ref(),
            Some(z.as_ref()),
            state.as_ref(),
            rocm_of(&acc_scratch),
            rocm_of(&out),
            NV,
            NK,
            D,
            EPS,
        )
        .expect("kda_gated_delta_rule_batched_into");
        dev.synchronize();
        trio_us += t0.elapsed().as_secs_f64() * 1e6;

        // Raw conv output must match the oracle's pre-SiLU sum too: the KDA
        // input is exactly this buffer, so a conv indexing error is otherwise
        // only visible through the KDA output.
        let got_conv = rocm_of(&conv_out).to_cpu_vec_f32().expect("read conv");
        let conv_worst = got_conv
            .iter()
            .zip(conv_raw.iter())
            .map(|(&g, &w)| (g as f64 - w as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(
            conv_worst <= 1e-4,
            "step {step}: short-conv raw output worst |err| {conv_worst:.3e} \
             (got {:?}, want {:?}); [channels,taps] vs [taps,channels] indexing \
             shows up here",
            &got_conv[..4.min(got_conv.len())],
            &conv_raw[..4.min(conv_raw.len())]
        );

        let got = rocm_of(&out).to_cpu_vec_f32().expect("read output");
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
            "step {step}: KDA output worst |err| {worst:.3e} at {worst_at} \
             (got {}, want {}); a chained-state, ring-shift, or gate error \
             shows up here",
            got[worst_at],
            want[worst_at]
        );
        eprintln!(
            "[kda-mega-trio] step {step}: conv Δ={conv_worst:.2e}, KDA Δ={worst:.2e}"
        );
    }

    // Convex state must have been carried and written, not recomputed from zero.
    let got_state = rocm_of(&state).to_cpu_vec_f32().expect("read state");
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

    eprintln!(
        "[kda-mega-trio] {STEPS} chained conv+KDA steps + state match f64 \
         (trio wall {trio_us:.1} us total, informational only — the persistent \
         megakernel has no KDA opcode, see header)"
    );
}
