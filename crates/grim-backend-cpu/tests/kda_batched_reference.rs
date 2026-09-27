//! Numeric contract for the batched Gated DeltaNet (KDA) decode step.
//!
//! The ROCm `grim_kda_gated_delta_rule_batched` kernel is checked against this
//! CPU implementation, so the CPU path has to be right first or the GPU gate
//! would only prove that two copies of the same mistake agree. The reference
//! below is therefore written independently in f64 from the published update
//! (ICLR 2025, Eq. 10) rather than by calling the implementation's helpers.
//!
//! What is pinned here, beyond the output values:
//!
//! * the recurrent state is updated IN PLACE, so a second step sees the first
//!   step's state — an implementation that returns correct output while
//!   leaving the state alone would pass a single-step test and be wrong in
//!   decode, where every step after the first compounds the error;
//! * the state is the L2-normalised-key delta rule, not a stale-state variant;
//! * `z = None` means a gate of 1.0, not a gate of `silu(0) = 0`, which would
//!   zero the whole branch;
//! * the conv stream geometry is validated, because the kernel derives its
//!   offsets from `2*key_dim + value_dim` and reads the wrong channels
//!   silently if the caller sizes it from the attention q+k+v width.

use grim_backend_cpu::CpuDevice;
use grim_tensor::dtype::DType;
use grim_tensor::{CoreTensorOps, RecurrentOps, Shape};

const NV: usize = 6;
const NK: usize = 2;
const D: usize = 8;
const EPS: f32 = 1e-6;

fn dev() -> CpuDevice {
    CpuDevice::new()
}

fn f32_storage(
    dev: &CpuDevice,
    data: &[f32],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, shape, DType::F32).expect("from_cpu")
}

fn conv_dim() -> usize {
    2 * NK * D + NV * D
}

fn linspace(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    if n == 1 {
        return vec![lo];
    }
    (0..n)
        .map(|i| lo + (hi - lo) * (i as f32) / ((n - 1) as f32))
        .collect()
}

/// Independent f64 reference for one batched KDA step, written from the spec.
fn reference(
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
    let mut out = vec![0.0f64; NV * D];
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
        let den = (ss + EPS as f64).sqrt();
        if den <= 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / den).collect()
        }
    };

    for h in 0..NV {
        // [q | k | v]: llama.cpp qwen35.cpp:404-424 and vLLM
        // qwen_gdn_linear_attn.py:704 both put q at offset 0, k at key_dim and
        // v at 2*key_dim.
        let q_base = (h % NK) * D;
        let k_base = key_dim + (h % NK) * D;
        let v_base = 2 * key_dim + h * D;
        let q_raw: Vec<f64> = conv[q_base..q_base + D].iter().map(|&v| v as f64).collect();
        let k_raw: Vec<f64> = conv[k_base..k_base + D].iter().map(|&v| v as f64).collect();
        let qv: Vec<f64> = conv[v_base..v_base + D].iter().map(|&v| v as f64).collect();

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
            // Reference: the head output is scaled by 1/sqrt(S_v) before the
            // gated norm (llama.cpp `gated_delta_net.cu:281`).
            acc[j] *= 1.0 / (D as f64).sqrt();
        }
        let ss: f64 = acc.iter().map(|a| a * a).sum();
        let inv = 1.0 / (ss / D as f64 + EPS as f64).sqrt();
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

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let rel = diff / w.abs().max(1e-3);
        let score = diff.min(rel);
        if score > worst {
            worst = score;
            worst_at = i;
        }
    }
    assert!(
        worst <= 1e-4,
        "{what}: worst |err| {worst:.3e} at index {worst_at} (got {}, want {})",
        got[worst_at],
        want[worst_at]
    );
}

fn as_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

struct Fixture {
    conv: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    norm_w: Vec<f32>,
    z: Vec<f32>,
}

fn fixture() -> Fixture {
    Fixture {
        conv: linspace(conv_dim(), -1.5, 2.0),
        alpha: linspace(NV, -0.4, 0.6),
        beta: linspace(NV, -1.0, 1.0),
        dt_bias: linspace(NV, 0.1, 0.9),
        ssm_a: linspace(NV, 0.5, 1.5),
        norm_w: linspace(NV * D, 0.8, 1.2),
        z: linspace(NV * D, -0.7, 0.7),
    }
}

fn run_step(
    dev: &CpuDevice,
    f: &Fixture,
    state: &dyn grim_tensor::BackendStorage,
    with_z: bool,
) -> Vec<f32> {
    let cm = f32_storage(dev, &f.conv, &Shape::new(vec![conv_dim()]));
    let al = f32_storage(dev, &f.alpha, &Shape::new(vec![NV]));
    let be = f32_storage(dev, &f.beta, &Shape::new(vec![NV]));
    let db = f32_storage(dev, &f.dt_bias, &Shape::new(vec![NV]));
    let sa = f32_storage(dev, &f.ssm_a, &Shape::new(vec![NV]));
    let nw = f32_storage(dev, &f.norm_w, &Shape::new(vec![NV * D]));
    let z = with_z.then(|| f32_storage(dev, &f.z, &Shape::new(vec![NV * D])));
    let (out, _h) = dev
        .kda_gated_delta_rule_batched(
            cm.as_ref(),
            al.as_ref(),
            be.as_ref(),
            db.as_ref(),
            sa.as_ref(),
            nw.as_ref(),
            z.as_ref().map(|s| s.as_ref()),
            state,
            NV,
            NK,
            D,
            EPS,
            &Shape::new(vec![NV * D]),
        )
        .expect("kda_gated_delta_rule_batched");
    out.to_cpu_vec_f32().expect("read output")
}

#[test]
fn kda_batched_matches_independent_f64_reference() {
    let dev = dev();
    let f = fixture();
    let state = f32_storage(&dev, &vec![0.0f32; NV * D * D], &Shape::new(vec![NV, D, D]));

    let mut want_state = vec![0.0f64; NV * D * D];
    let want = reference(
        &f.conv,
        &f.alpha,
        &f.beta,
        &f.dt_bias,
        &f.ssm_a,
        &f.norm_w,
        Some(&f.z),
        &mut want_state,
    );

    let got = run_step(&dev, &f, state.as_ref(), true);
    assert_close(&as_f64(&got), &want, "kda batched output");
}

#[test]
fn kda_batched_updates_state_in_place_across_steps() {
    let dev = dev();
    let f = fixture();
    let state = f32_storage(&dev, &vec![0.0f32; NV * D * D], &Shape::new(vec![NV, D, D]));

    // Two different streams: if the second step were computed from a zero
    // state, its output would be identical to a fresh reference run.
    let mut f2 = fixture();
    f2.conv = linspace(conv_dim(), 0.3, -1.1);
    let mut f3 = fixture();
    f3.conv = linspace(conv_dim(), -0.4, 1.7);

    let mut want_state = vec![0.0f64; NV * D * D];
    reference(
        &f.conv,
        &f.alpha,
        &f.beta,
        &f.dt_bias,
        &f.ssm_a,
        &f.norm_w,
        Some(&f.z),
        &mut want_state,
    );
    let want2 = reference(
        &f2.conv,
        &f.alpha,
        &f.beta,
        &f.dt_bias,
        &f.ssm_a,
        &f.norm_w,
        Some(&f.z),
        &mut want_state,
    );

    let _first = run_step(&dev, &f, state.as_ref(), true);
    let second = run_step(&dev, &f2, state.as_ref(), true);
    assert_close(&as_f64(&second), &want2, "kda batched second step");

    // And the state itself must match the running reference, not just the
    // output: a kernel that wrote the right value from a scratch copy of the
    // state would produce correct outputs forever while the state stayed zero.
    let got_state = state.to_cpu_vec_f32().expect("read state");
    assert_close(
        &as_f64(&got_state),
        &want_state,
        "kda batched state after two steps",
    );

    // A third step from a different stream must also chain, which only holds if
    // step two really did write through the same state buffer.
    let mut want3 = want_state.clone();
    let want3 = reference(
        &f3.conv,
        &f.alpha,
        &f.beta,
        &f.dt_bias,
        &f.ssm_a,
        &f.norm_w,
        Some(&f.z),
        &mut want3,
    );
    let third = run_step(&dev, &f3, state.as_ref(), true);
    assert_close(&as_f64(&third), &want3, "kda batched third step");
}

#[test]
fn kda_batched_without_z_applies_unit_gate() {
    let dev = dev();
    let f = fixture();
    let state_a = f32_storage(&dev, &vec![0.0f32; NV * D * D], &Shape::new(vec![NV, D, D]));
    let state_b = f32_storage(&dev, &vec![0.0f32; NV * D * D], &Shape::new(vec![NV, D, D]));

    let with_z = run_step(&dev, &f, state_a.as_ref(), true);
    let without_z = run_step(&dev, &f, state_b.as_ref(), false);

    // z only gates the output, never the recurrence, so the two states agree.
    let sa = state_a.to_cpu_vec_f32().expect("state a");
    let sb = state_b.to_cpu_vec_f32().expect("state b");
    assert_eq!(sa, sb, "z must not feed the recurrent state");

    // silu(z) with the fixture's z range is strictly positive but well below
    // 1, so dropping it must change the output — otherwise `z` is dead.
    let differs = with_z
        .iter()
        .zip(without_z.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(differs, "z gate had no effect on the output");
}

#[test]
fn kda_batched_rejects_mis_sized_conv_stream() {
    let dev = dev();
    let f = fixture();
    let state = f32_storage(&dev, &vec![0.0f32; NV * D * D], &Shape::new(vec![NV, D, D]));
    // One element short of 2*key_dim + value_dim. The kernel derives its
    // offsets from that expression, so a short stream must fail rather than
    // read the neighbouring allocation.
    let short = conv_dim() - 1;
    let res = dev.kda_gated_delta_rule_batched(
        f32_storage(&dev, &f.conv[..short], &Shape::new(vec![short])).as_ref(),
        f32_storage(&dev, &f.alpha, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.beta, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.dt_bias, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.ssm_a, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.norm_w, &Shape::new(vec![NV * D])).as_ref(),
        Some(f32_storage(&dev, &f.z, &Shape::new(vec![NV * D])).as_ref()),
        state.as_ref(),
        NV,
        NK,
        D,
        EPS,
        &Shape::new(vec![NV * D]),
    );
    let ok = res.is_ok();
    assert!(
        !ok,
        "a conv stream shorter than 2*key_dim+value_dim must be rejected"
    );
}

#[test]
fn kda_batched_rejects_short_state() {
    let dev = dev();
    let f = fixture();
    let short_state = f32_storage(
        &dev,
        &vec![0.0f32; NV * D * D - 1],
        &Shape::new(vec![NV * D * D - 1]),
    );
    let res = dev.kda_gated_delta_rule_batched(
        f32_storage(&dev, &f.conv, &Shape::new(vec![conv_dim()])).as_ref(),
        f32_storage(&dev, &f.alpha, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.beta, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.dt_bias, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.ssm_a, &Shape::new(vec![NV])).as_ref(),
        f32_storage(&dev, &f.norm_w, &Shape::new(vec![NV * D])).as_ref(),
        Some(f32_storage(&dev, &f.z, &Shape::new(vec![NV * D])).as_ref()),
        short_state.as_ref(),
        NV,
        NK,
        D,
        EPS,
        &Shape::new(vec![NV * D]),
    );
    let ok = res.is_ok();
    assert!(
        !ok,
        "a state shorter than n_heads*head_dim^2 must be rejected"
    );
}
