//! Task 10.5 gate: BitNet b1.58 GEMM parity — GPU `grim_bitnet_gemm_w158a8`
//! (f32 activations) and `grim_bitnet_gemm_w158a8_int8` (int8 activations)
//! against a CPU ternary reference over the plan's shape matrix
//! (M ∈ {1, 16}, N = 2048, K = 2048).
//!
//! Weight layout (both kernels): `B_ternary` is `[N, K/4]`, 2-bit packed,
//! code `01` = +1, `10` = −1, `00` = 0; `scale_b` is per-output-channel.
//! Tolerances per plan: relative error < 2e-2 for the int8 activation arm,
//! < 1e-5 for the f32 activation arm (int accumulation is exact; only the
//! final f32 scale multiply rounds).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, MemoryOps, Shape, dtype::DType};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed"))
        .ok()
}

/// Deterministic pseudo-ternary weight blob for one channel: ~40% zeros,
/// +1/−1 split evenly, packed 2 bits per weight (code 1=+1, 2=−1).
fn pack_ternary_row(seed: u64, k: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut packed = vec![0u8; k / 4];
    for w in 0..k {
        let r = (next() >> 33) % 10;
        let code: u8 = match r {
            0..=2 => 1, // +1
            3..=5 => 2, // −1
            _ => 0,     // 0
        };
        packed[w / 4] |= code << ((w % 4) * 2);
    }
    packed
}

/// CPU reference: C[m,n] = scale_a * scale_b[n] * Σ_k w[k] * A[m,k].
fn cpu_reference(
    a: &[f32],
    a_int8: &[i8],
    packed: &[u8],
    scale_b: &[f32],
    scale_a: f32,
    m: usize,
    n: usize,
    k: usize,
    use_int8: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0i64;
            for ki in 0..k {
                let code = (packed[ni * (k / 4) + ki / 4] >> ((ki % 4) * 2)) & 0x3;
                let w: i64 = match code {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                if w == 0 {
                    continue;
                }
                let av = if use_int8 {
                    a_int8[mi * k + ki] as i64
                } else {
                    // f32 arm multiplies in f32; accumulate in f32 via the
                    // same order the kernel uses (sequential over k).
                    return cpu_reference_f32(a, packed, scale_b, scale_a, m, n, k);
                };
                acc += w * av;
            }
            out[mi * n + ni] = (acc as f32) * scale_a * scale_b[ni];
        }
    }
    out
}

fn cpu_reference_f32(
    a: &[f32],
    packed: &[u8],
    scale_b: &[f32],
    scale_a: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let code = (packed[ni * (k / 4) + ki / 4] >> ((ki % 4) * 2)) & 0x3;
                match code {
                    1 => acc += a[mi * k + ki],
                    2 => acc -= a[mi * k + ki],
                    _ => {}
                }
            }
            out[mi * n + ni] = acc * scale_a * scale_b[ni];
        }
    }
    out
}

fn run_arm(dev: &RocmDevice, m: usize, n: usize, k: usize, use_int8: bool) -> TestResult {
    // Activations: deterministic f32, then absmax-scaled int8 for the int8 arm.
    let a_f32: Vec<f32> = (0..m * k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect();
    let absmax = a_f32.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale_a = if use_int8 { absmax / 127.0 } else { 1.0 };
    let a_int8: Vec<i8> = a_f32
        .iter()
        .map(|&v| (v / scale_a).round().clamp(-127.0, 127.0) as i8)
        .collect();

    // Weights: one packed row per output channel + per-channel scales.
    let mut packed = Vec::with_capacity(n * (k / 4));
    let scale_b: Vec<f32> = (0..n).map(|i| 0.02 + (i % 13) as f32 * 0.01).collect();
    for ni in 0..n {
        packed.extend(pack_ternary_row((ni as u64) + 1, k));
    }

    let expected = cpu_reference(
        &a_f32,
        &a_int8,
        &packed,
        &scale_b,
        scale_a,
        m,
        n,
        k,
        use_int8,
    );

    let a_dev = if use_int8 {
        let bytes: Vec<u8> = a_int8.iter().map(|&x| x as u8).collect();
        MemoryOps::from_cpu_bytes(dev, &bytes, &Shape::from_slice(&[m, k]), DType::U8)?
    } else {
        CoreTensorOps::from_cpu(dev, &a_f32, &Shape::from_slice(&[m, k]), DType::F32)?
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        dev,
        &packed,
        &Shape::from_slice(&[n * (k / 4)]),
        DType::U8,
    )?;
    let sb_dev = CoreTensorOps::from_cpu(
        dev,
        &scale_b,
        &Shape::from_slice(&[n]),
        DType::F32,
    )?;
    let out_dev = CoreTensorOps::zeros(dev, &Shape::from_slice(&[m, n]), DType::F32)?;

    let a_s = grim_backend_rocm::device::util::as_rocm(a_dev.as_ref())?;
    let b_s = grim_backend_rocm::device::util::as_rocm(b_dev.as_ref())?;
    let sb_s = grim_backend_rocm::device::util::as_rocm(sb_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    if use_int8 {
        dev.launch_bitnet_gemm_int8(a_s, b_s, sb_s, out_s, m, n, k, scale_a)?;
    } else {
        dev.launch_bitnet_gemm_w158a8(a_s, b_s, sb_s, out_s, m, n, k, scale_a)?;
    }
    dev.synchronize();

    let actual = out_dev.to_cpu_vec_f32()?;
    let tol = if use_int8 { 2e-2 } else { 1e-5 };
    let mut worst = 0.0f32;
    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let denom = exp.abs().max(1e-6);
        let rel = ((act - exp).abs() / denom) as f64;
        worst = worst.max(rel as f32);
        assert!(
            rel < tol,
            "{arm} M={m} mismatch at [{i}]: actual={act}, expected={exp}, rel={rel}",
            arm = if use_int8 { "int8" } else { "f32" },
        );
    }
    println!("bitnet {arm} M={m} N={n} K={k}: worst rel err {worst:.2e}", arm = if use_int8 { "int8" } else { "f32" });
    Ok(())
}

#[test]
#[ignore]
fn bitnet_w158a8_f32_and_int8_parity_m1_m16() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    for &m in &[1usize, 16usize] {
        run_arm(&dev, m, 2048, 2048, false)?;
        run_arm(&dev, m, 2048, 2048, true)?;
    }
    Ok(())
}
