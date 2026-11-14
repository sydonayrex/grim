use grim_backend_rocm::kernels::compute_kernels::OTHER_KERNEL_SOURCE;

/// CPU oracle for per-head RMSNorm + partial NeoX/interleaved RoPE over head_dim ∈ {64, 128, 256}.
fn cpu_qk_rope_wide_oracle(
    x: &[f32],
    gamma: &[f32],
    pos: u32,
    d: usize,
    rotary_dim: usize,
    base: f32,
    eps: f32,
    interleaved: bool,
) -> Vec<f32> {
    assert_eq!(x.len(), d);
    assert_eq!(gamma.len(), d);
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0f32 / (ss / (d as f32) + eps).sqrt();
    let normed: Vec<f32> = x
        .iter()
        .zip(gamma.iter())
        .map(|(&v, &g)| v * inv_rms * g)
        .collect();
    let mut out = normed.clone();
    let r_half = rotary_dim / 2;
    for i in 0..r_half {
        let freq = 1.0f32 / base.powf((2 * i) as f32 / (rotary_dim as f32));
        let angle = (pos as f32) * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        let (a_idx, b_idx) = if interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, r_half + i)
        };
        let x1 = normed[a_idx];
        let x2 = normed[b_idx];
        out[a_idx] = x1 * cos_v - x2 * sin_v;
        out[b_idx] = x2 * cos_v + x1 * sin_v;
    }
    out
}

/// Ported from Radiance `libr4d/r4d_selftest.cpp:2642-2656` & `tests/yarn_test.cpp:38-51`:
/// Exact double-precision vLLM static YaRN cos/sin oracle at rotary widths {64, 128, 256}.
fn radiance_yarn_ref(pos: f64, i: usize, rot: usize, theta: f64, factor: f64) -> (f64, f64, f64) {
    let (low, high) = match rot {
        64 => (14.0, 22.0),
        128 => (28.0, 43.0),
        256 => (56.0, 85.0),
        _ => panic!("unsupported rot {rot}"),
    };
    let magnitude = 1.0 + 0.1 * factor.ln();
    let inv = theta.powf(-2.0 * (i as f64) / (rot as f64));
    let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
    let angle = pos * (inv / factor * ramp + inv * (1.0 - ramp));
    let tol = magnitude * (angle.abs() * f64::exp2(-20.0) + f64::exp2(-21.0));
    (angle.cos() * magnitude, angle.sin() * magnitude, tol)
}

#[test]
fn compute_kernels_contains_wide_qk_rope() {
    assert!(
        OTHER_KERNEL_SOURCE.contains("grim_qk_rope_wide_dev_base"),
        "missing grim_qk_rope_wide_dev_base in OTHER_KERNEL_SOURCE"
    );
}

#[test]
fn radiance_oracle_partial_rotary_and_closed_form_identities() {
    // 1. Ported from `ref_test.cpp:4615` (`op_qk_norm_rope`):
    // At pos = 0, rotation is identity, isolating per-head RMSNorm: [3, 4] -> rms = sqrt(12.5) = 3.5355339.
    let mut x64 = vec![0.0f32; 64];
    x64[0] = 3.0;
    x64[1] = 4.0;
    let g64 = vec![2.0f32; 64];
    let out0 = cpu_qk_rope_wide_oracle(&x64, &g64, 0, 64, 64, 10000.0, 0.0, false);
    let rms64 = ((9.0f32 + 16.0f32) / 64.0f32).sqrt();
    assert!((out0[0] - 2.0 * 3.0 / rms64).abs() < 1e-6);
    assert!((out0[1] - 2.0 * 4.0 / rms64).abs() < 1e-6);

    // 2. Partial rotary (rotary_dim = 64 of head_dim = 256, Qwen3.5/GLM-5 geometry):
    // Channels 0..64 rotate in NeoX pairs (i, i+32); channels 64..256 are RMSNormed only.
    let d = 256;
    let rotary_dim = 64;
    let x: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let gamma: Vec<f32> = (0..d).map(|i| 0.5 + (i as f32) * 0.002).collect();
    let out = cpu_qk_rope_wide_oracle(&x, &gamma, 7, d, rotary_dim, 10_000_000.0, 1e-6, false);
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0f32 / (ss / (d as f32) + 1e-6).sqrt();
    for tail_i in rotary_dim..d {
        let want = x[tail_i] * inv_rms * gamma[tail_i];
        assert!((out[tail_i] - want).abs() < 1e-6);
    }

    // 3. Ported from `r4d_selftest.cpp:2654, 2795-2809`:
    // Verify f32 YaRN evaluation against double-precision `radiance_yarn_ref` across 512K context positions.
    let yarn_positions: [u32; 7] = [0, 1, 4096, 262143, 262144, 500000, 524287];
    for &rot in &[64usize, 128, 256] {
        let (low, high) = match rot {
            64 => (14.0f32, 22.0f32),
            128 => (28.0f32, 43.0f32),
            _ => (56.0f32, 85.0f32),
        };
        let mag = 1.0f32 + 0.1f32 * 2.0f32.ln();
        for &p in &yarn_positions {
            for i in 0..(rot / 2) {
                let inv = 10_000_000.0f32.powf(-2.0f32 * (i as f32) / (rot as f32));
                let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                let a = (p as f32) * (inv / 2.0f32 * ramp + inv * (1.0f32 - ramp));
                let (s32, c32) = (a.sin() * mag, a.cos() * mag);
                let (c64, s64, tol) = radiance_yarn_ref(p as f64, i, rot, 1e7, 2.0);
                assert!((c32 as f64 - c64).abs() <= tol);
                assert!((s32 as f64 - s64).abs() <= tol);
            }
        }
    }
}
