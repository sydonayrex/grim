//! Comprehensive tests for MXFP4 GEMM, Backward GEMM, and Fused RMSNorm + MXFP4 + RoPE + KV Cache.

use grim_backend_rocm::RocmDevice;
use grim_quant::{f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};
use grim_tensor::{
    BackendDevice, Shape,
    dtype::{ArithType, DType, FloatPackScheme, Storage},
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    match panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    }) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

fn as_u8_slice<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

fn cpu_rmsnorm(x: &[f32], gamma: &[f32], m: usize, k: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    for row in 0..m {
        let mut sum_sq = 0.0f32;
        for col in 0..k {
            let v = x[row * k + col];
            sum_sq += v * v;
        }
        let rms = 1.0f32 / (sum_sq / k as f32 + eps).sqrt();
        for col in 0..k {
            out[row * k + col] = x[row * k + col] * rms * gamma[col];
        }
    }
    out
}

fn cpu_rope_in_place(
    x: &mut [f32],
    m: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    positions: &[u32],
) {
    for row in 0..m {
        let pos = positions[row] as f32;
        for h in 0..num_heads {
            let base = row * (num_heads * head_dim) + h * head_dim;
            for i in 0..(rotary_dim / 2) {
                let freq = 1.0f32 / theta.powf((2.0 * i as f32) / (rotary_dim as f32));
                let angle = pos * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;

                let v0 = x[idx0];
                let v1 = x[idx1];

                x[idx0] = v0 * cos_a - v1 * sin_a;
                x[idx1] = v0 * sin_a + v1 * cos_a;
            }
        }
    }
}

/// Build the YaRN-ramp-corrected inverse-frequency table, mirroring
/// `RocmDevice::rope_launch_yarn` in roc_device.rs.
fn cpu_yarn_inv_freq(
    rotary_half: usize,
    head_dim: usize,
    theta: f32,
    factor: f32,
    original_max_pos: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    (0..rotary_half)
        .map(|i| {
            let freq = 1.0f32 / theta.powf((2.0 * i as f32) / (head_dim as f32));
            let wavelength = 2.0 * std::f32::consts::PI / freq;
            let low = original_max_pos / beta_slow;
            let high = original_max_pos / beta_fast;
            if wavelength < high {
                freq
            } else if wavelength > low {
                freq / factor
            } else {
                let ramp = (original_max_pos / wavelength - beta_slow) / (beta_fast - beta_slow);
                (1.0 - ramp) * (freq / factor) + ramp * freq
            }
        })
        .collect()
}

/// YaRN RoPE on the CPU, mirroring `grim_rope_yarn` (mscale applied to sin/cos).
fn cpu_rope_yarn_in_place(
    x: &mut [f32],
    m: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &[f32],
    mscale: f32,
    positions: &[u32],
) {
    let rotary_half = rotary_dim / 2;
    for row in 0..m {
        let pos = positions[row] as f32;
        for h in 0..num_heads {
            let base = row * (num_heads * head_dim) + h * head_dim;
            for i in 0..rotary_half {
                let angle = pos * inv_freq[i];
                let cos_a = angle.cos() * mscale;
                let sin_a = angle.sin() * mscale;

                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;

                let v0 = x[idx0];
                let v1 = x[idx1];

                x[idx0] = v0 * cos_a - v1 * sin_a;
                x[idx1] = v0 * sin_a + v1 * cos_a;
            }
        }
    }
}

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_mxfp4_gemm_kernel_source_compiles() {
    let dev = RocmDevice::new(0);
    let kernel_src = grim_backend_rocm::kernels::source_asm::compute_kernel_source();
    assert!(kernel_src.contains("grim_fused_rmsnorm_mxfp4_gemm_rope_kv"));
    assert!(kernel_src.contains("grim_fused_rmsnorm_mxfp4_gemm"));
    assert!(kernel_src.contains("grim_mxfp4_gemm_tiled"));
    assert!(kernel_src.contains("grim_mxfp4_backward_gemm"));
    assert_eq!(dev.wavefront_size(), grim_backend_rocm::WavefrontSize::W32);
}

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_mxfp4_tiled_gemm_parity_against_cpu() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (4usize, 128usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();

    let shared_exp = 127u8;
    let b_orig: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.03).cos() * 1.5).collect();
    let b_codes: Vec<u8> = b_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let b_dequant: Vec<f32> = b_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

    // CPU Oracle: C = A @ B_dequant^T
    let mut expected_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant[ni * k + ki];
            }
            expected_c[mi * n + ni] = sum;
        }
    }

    let mut b_packed = vec![0u8; (n * k) / 2];
    for j in 0..(n * k) {
        let code = b_codes[j] & 0x0F;
        if j % 2 == 0 {
            b_packed[j / 2] |= code;
        } else {
            b_packed[j / 2] |= code << 4;
        }
    }
    let num_blocks = (n * k) / 32;
    let b_exps_u8: Vec<u8> = vec![shared_exp; num_blocks];

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[n, k]);
    let exps_shape = Shape::from_slice(&[num_blocks]);

    let a_dev = BackendDevice::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
    let b_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &b_packed,
        &b_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        },
    )?;
    let exps_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &b_exps_u8,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;

    let out_shape = Shape::from_slice(&[m, n]);
    let out_dev = BackendDevice::zeros(&dev, &out_shape, DType::F32)?;

    let a_s = grim_backend_rocm::device::util::as_rocm(a_dev.as_ref())?;
    let b_s = grim_backend_rocm::device::util::as_rocm(b_dev.as_ref())?;
    let exps_s = grim_backend_rocm::device::util::as_rocm(exps_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    dev.launch_mxfp4_gemm_tiled(
        a_s,
        b_s.device_ptr_u64().ok_or("b codes ptr")?,
        exps_s.device_ptr_u64().ok_or("exps ptr")?,
        out_s,
        m,
        n,
        k,
    )?;
    dev.synchronize();

    let actual_c = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual_c.len(), expected_c.len());

    for (i, (&act, &exp)) in actual_c.iter().zip(expected_c.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "Mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_fused_rmsnorm_mxfp4_gemm_parity_against_cpu() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (2usize, 64usize, 32usize);
    let eps = 1e-5f32;

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() + 0.5).collect();
    let gamma_data: Vec<f32> = (0..k).map(|i| (i as f32 * 0.05).cos() + 1.0).collect();

    let shared_exp = 126u8;
    let w_orig: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.04).sin() * 2.0).collect();
    let w_codes: Vec<u8> = w_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let w_dequant: Vec<f32> = w_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

    // Step 1: CPU RMSNorm
    let x_norm = cpu_rmsnorm(&x_data, &gamma_data, m, k, eps);

    // Step 2: CPU GEMM
    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[ni * k + ki];
            }
            expected[mi * n + ni] = sum;
        }
    }

    let mut w_packed = vec![0u8; (n * k) / 2];
    for j in 0..(n * k) {
        let code = w_codes[j] & 0x0F;
        if j % 2 == 0 {
            w_packed[j / 2] |= code;
        } else {
            w_packed[j / 2] |= code << 4;
        }
    }
    let num_blocks = (n * k) / 32;
    let w_exps: Vec<u8> = vec![shared_exp; num_blocks];

    let x_shape = Shape::from_slice(&[m, k]);
    let gamma_shape = Shape::from_slice(&[k]);
    let w_shape = Shape::from_slice(&[n, k]);
    let exps_shape = Shape::from_slice(&[num_blocks]);

    let x_dev = BackendDevice::from_cpu(&dev, &x_data, &x_shape, DType::F32)?;
    let gamma_dev = BackendDevice::from_cpu(&dev, &gamma_data, &gamma_shape, DType::F32)?;
    let w_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_packed,
        &w_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        },
    )?;
    let exps_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_exps,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;

    let (out_dev, handle) = dev.fused_rmsnorm_mxfp4_gemm(
        x_dev.as_ref(),
        gamma_dev.as_ref(),
        w_dev.as_ref(),
        exps_dev.as_ref(),
        m,
        n,
        k,
        eps,
    )?;
    handle.synchronize()?;

    let actual = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual.len(), expected.len());

    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "Fused RMSNorm MXFP4 mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_fused_rmsnorm_mxfp4_gemm_rope_kv_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 64usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 16usize;
    let rotary_dim = 16usize;
    let theta = 10000.0f32;
    let eps = 1e-5f32;
    let max_seq_len = 16usize;

    let n_q = num_q_heads * head_dim;
    let n_k = num_kv_heads * head_dim;
    let n_v = num_kv_heads * head_dim;
    let n_total = n_q + n_k + n_v; // 32 + 16 + 16 = 64

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.08).sin()).collect();
    let gamma_data: Vec<f32> = vec![1.0f32; k];
    let positions: Vec<u32> = vec![0, 1];

    let shared_exp = 127u8;
    let w_orig: Vec<f32> = (0..n_total * k)
        .map(|i| (i as f32 * 0.02).cos() * 1.2)
        .collect();
    let w_codes: Vec<u8> = w_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let w_dequant: Vec<f32> = w_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

    // CPU Reference Pipeline:
    // 1. RMSNorm
    let x_norm = cpu_rmsnorm(&x_data, &gamma_data, m, k, eps);

    // 2. GEMM to Q, K, V
    let mut q_expected = vec![0.0f32; m * n_q];
    let mut k_expected = vec![0.0f32; m * n_k];
    let mut v_expected = vec![0.0f32; m * n_v];

    for mi in 0..m {
        for ni in 0..n_q {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[ni * k + ki];
            }
            q_expected[mi * n_q + ni] = sum;
        }
        for ni in 0..n_k {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[(n_q + ni) * k + ki];
            }
            k_expected[mi * n_k + ni] = sum;
        }
        for ni in 0..n_v {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[(n_q + n_k + ni) * k + ki];
            }
            v_expected[mi * n_v + ni] = sum;
        }
    }

    // 3. RoPE on Q and K
    cpu_rope_in_place(
        &mut q_expected,
        m,
        num_q_heads,
        head_dim,
        rotary_dim,
        theta,
        &positions,
    );
    cpu_rope_in_place(
        &mut k_expected,
        m,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
        &positions,
    );

    // Prepare GPU tensors
    let mut w_packed = vec![0u8; (n_total * k) / 2];
    for j in 0..(n_total * k) {
        let code = w_codes[j] & 0x0F;
        if j % 2 == 0 {
            w_packed[j / 2] |= code;
        } else {
            w_packed[j / 2] |= code << 4;
        }
    }
    let num_blocks = (n_total * k) / 32;
    let w_exps: Vec<u8> = vec![shared_exp; num_blocks];

    let x_shape = Shape::from_slice(&[m, k]);
    let gamma_shape = Shape::from_slice(&[k]);
    let w_shape = Shape::from_slice(&[n_total, k]);
    let exps_shape = Shape::from_slice(&[num_blocks]);
    let q_shape = Shape::from_slice(&[m, n_q]);
    let kv_cache_shape = Shape::from_slice(&[max_seq_len, n_k]);
    let pos_shape = Shape::from_slice(&[m]);

    let x_dev = BackendDevice::from_cpu(&dev, &x_data, &x_shape, DType::F32)?;
    let gamma_dev = BackendDevice::from_cpu(&dev, &gamma_data, &gamma_shape, DType::F32)?;
    let w_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_packed,
        &w_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        },
    )?;
    let exps_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_exps,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;
    let pos_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&positions), &pos_shape, DType::U32)?;

    let q_out_dev = BackendDevice::zeros(&dev, &q_shape, DType::F32)?;
    let k_cache_dev = BackendDevice::zeros(&dev, &kv_cache_shape, DType::F32)?;
    let v_cache_dev = BackendDevice::zeros(&dev, &kv_cache_shape, DType::F32)?;

    let handle = dev.fused_rmsnorm_mxfp4_gemm_rope_kv(
        x_dev.as_ref(),
        gamma_dev.as_ref(),
        w_dev.as_ref(),
        exps_dev.as_ref(),
        Some(q_out_dev.as_ref()),
        Some(k_cache_dev.as_ref()),
        Some(v_cache_dev.as_ref()),
        None,
        Some(pos_dev.as_ref()),
        m,
        k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
        None,
        1.0f32,
        eps,
        max_seq_len,
    )?;
    handle.synchronize()?;

    let q_actual = q_out_dev.to_cpu_vec_f32()?;
    let k_cache_actual = k_cache_dev.to_cpu_vec_f32()?;
    let v_cache_actual = v_cache_dev.to_cpu_vec_f32()?;

    // Check Q parity
    for (i, (&act, &exp)) in q_actual.iter().zip(q_expected.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "Q mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    // Check K Cache for position 0 and 1
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_k {
            let act = k_cache_actual[pos * n_k + c];
            let exp = k_expected[row * n_k + c];
            let err = (act - exp).abs();
            assert!(
                err < 1e-4,
                "K Cache mismatch at pos {pos}, col {c}: actual={act}, expected={exp}, err={err}"
            );
        }
    }

    // Check V Cache for position 0 and 1
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_v {
            let act = v_cache_actual[pos * n_v + c];
            let exp = v_expected[row * n_v + c];
            let err = (act - exp).abs();
            assert!(
                err < 1e-4,
                "V Cache mismatch at pos {pos}, col {c}: actual={act}, expected={exp}, err={err}"
            );
        }
    }

    Ok(())
}

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_fused_rmsnorm_mxfp4_gemm_rope_kv_yarn_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 64usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 16usize;
    let rotary_dim = 16usize;
    let theta = 10000.0f32;
    let eps = 1e-5f32;
    let max_seq_len = 16usize;

    // YaRN parameters
    let yarn_factor = 2.0f32;
    let original_max_pos = 32.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let attention_factor = 1.0f32;
    let mscale = attention_factor;

    let n_q = num_q_heads * head_dim;
    let n_k = num_kv_heads * head_dim;
    let n_v = num_kv_heads * head_dim;
    let n_total = n_q + n_k + n_v;

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.08).sin()).collect();
    let gamma_data: Vec<f32> = vec![1.0f32; k];
    let positions: Vec<u32> = vec![0, 1];

    let shared_exp = 127u8;
    let w_orig: Vec<f32> = (0..n_total * k)
        .map(|i| (i as f32 * 0.02).cos() * 1.2)
        .collect();
    let w_codes: Vec<u8> = w_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let w_dequant: Vec<f32> = w_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

    // CPU Reference Pipeline (with YaRN):
    // 1. RMSNorm
    let x_norm = cpu_rmsnorm(&x_data, &gamma_data, m, k, eps);

    // 2. GEMM to Q, K, V
    let mut q_expected = vec![0.0f32; m * n_q];
    let mut k_expected = vec![0.0f32; m * n_k];
    let mut v_expected = vec![0.0f32; m * n_v];
    for mi in 0..m {
        for ni in 0..n_q {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[ni * k + ki];
            }
            q_expected[mi * n_q + ni] = sum;
        }
        for ni in 0..n_k {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[(n_q + ni) * k + ki];
            }
            k_expected[mi * n_k + ni] = sum;
        }
        for ni in 0..n_v {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_norm[mi * k + ki] * w_dequant[(n_q + n_k + ni) * k + ki];
            }
            v_expected[mi * n_v + ni] = sum;
        }
    }

    // 3. YaRN RoPE on Q and K
    let rotary_half = rotary_dim / 2;
    let inv_freq = cpu_yarn_inv_freq(
        rotary_half,
        head_dim,
        theta,
        yarn_factor,
        original_max_pos,
        beta_fast,
        beta_slow,
    );
    cpu_rope_yarn_in_place(
        &mut q_expected,
        m,
        num_q_heads,
        head_dim,
        rotary_dim,
        &inv_freq,
        mscale,
        &positions,
    );
    cpu_rope_yarn_in_place(
        &mut k_expected,
        m,
        num_kv_heads,
        head_dim,
        rotary_dim,
        &inv_freq,
        mscale,
        &positions,
    );

    // Prepare GPU tensors
    let mut w_packed = vec![0u8; (n_total * k) / 2];
    for j in 0..(n_total * k) {
        let code = w_codes[j] & 0x0F;
        if j % 2 == 0 {
            w_packed[j / 2] |= code;
        } else {
            w_packed[j / 2] |= code << 4;
        }
    }
    let num_blocks = (n_total * k) / 32;
    let w_exps: Vec<u8> = vec![shared_exp; num_blocks];

    let x_shape = Shape::from_slice(&[m, k]);
    let gamma_shape = Shape::from_slice(&[k]);
    let w_shape = Shape::from_slice(&[n_total, k]);
    let exps_shape = Shape::from_slice(&[num_blocks]);
    let inv_freq_shape = Shape::from_slice(&[rotary_half]);
    let q_shape = Shape::from_slice(&[m, n_q]);
    let kv_cache_shape = Shape::from_slice(&[max_seq_len, n_k]);
    let pos_shape = Shape::from_slice(&[m]);

    let x_dev = BackendDevice::from_cpu(&dev, &x_data, &x_shape, DType::F32)?;
    let gamma_dev = BackendDevice::from_cpu(&dev, &gamma_data, &gamma_shape, DType::F32)?;
    let w_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_packed,
        &w_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        },
    )?;
    let exps_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_exps,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;
    let inv_freq_dev = BackendDevice::from_cpu(&dev, &inv_freq, &inv_freq_shape, DType::F32)?;
    let pos_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&positions), &pos_shape, DType::U32)?;

    let q_out_dev = BackendDevice::zeros(&dev, &q_shape, DType::F32)?;
    let k_cache_dev = BackendDevice::zeros(&dev, &kv_cache_shape, DType::F32)?;
    let v_cache_dev = BackendDevice::zeros(&dev, &kv_cache_shape, DType::F32)?;

    let handle = dev.fused_rmsnorm_mxfp4_gemm_rope_kv(
        x_dev.as_ref(),
        gamma_dev.as_ref(),
        w_dev.as_ref(),
        exps_dev.as_ref(),
        Some(q_out_dev.as_ref()),
        Some(k_cache_dev.as_ref()),
        Some(v_cache_dev.as_ref()),
        None,
        Some(pos_dev.as_ref()),
        m,
        k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
        Some(inv_freq_dev.as_ref()),
        mscale,
        eps,
        max_seq_len,
    )?;
    handle.synchronize()?;

    let q_actual = q_out_dev.to_cpu_vec_f32()?;
    let k_cache_actual = k_cache_dev.to_cpu_vec_f32()?;
    let v_cache_actual = v_cache_dev.to_cpu_vec_f32()?;

    for (i, (&act, &exp)) in q_actual.iter().zip(q_expected.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-3,
            "Q mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_k {
            let act = k_cache_actual[pos * n_k + c];
            let exp = k_expected[row * n_k + c];
            let err = (act - exp).abs();
            assert!(
                err < 1e-3,
                "K Cache mismatch at pos {pos}, col {c}: actual={act}, expected={exp}, err={err}"
            );
        }
    }
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_v {
            let act = v_cache_actual[pos * n_v + c];
            let exp = v_expected[row * n_v + c];
            let err = (act - exp).abs();
            assert!(
                err < 1e-3,
                "V Cache mismatch at pos {pos}, col {c}: actual={act}, expected={exp}, err={err}"
            );
        }
    }

    Ok(())
}

/// LFM2-style fused QKV: MXFP4 GEMM -> per-head QK-Norm -> RoPE (plain theta).
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_fused_mxfp4_gemm_qk_norm_rope_kv_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 64usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 16usize;
    let rotary_dim = 16usize;
    let theta = 10000.0f32;
    let eps = 1e-5f32;
    let max_seq_len = 16usize;

    let n_q = num_q_heads * head_dim;
    let n_k = num_kv_heads * head_dim;
    let n_v = num_kv_heads * head_dim;
    let n_total = n_q + n_k + n_v;

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.08).sin()).collect();
    let gamma_qk: Vec<f32> = vec![1.0f32; head_dim];
    let positions: Vec<u32> = vec![0, 1];

    let shared_exp = 127u8;
    let w_orig: Vec<f32> = (0..n_total * k)
        .map(|i| (i as f32 * 0.02).cos() * 1.2)
        .collect();
    let w_codes: Vec<u8> = w_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let w_dequant: Vec<f32> = w_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

    // CPU oracle: GEMM -> per-head QK-Norm -> RoPE
    let mut q_expected = vec![0.0f32; m * n_q];
    let mut k_expected = vec![0.0f32; m * n_k];
    let mut v_expected = vec![0.0f32; m * n_v];
    for mi in 0..m {
        let mut qkv = vec![0.0f32; n_total];
        for ni in 0..n_total {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_data[mi * k + ki] * w_dequant[ni * k + ki];
            }
            qkv[ni] = sum;
        }
        for h in 0..num_q_heads {
            let base = h * head_dim;
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                ss += qkv[base + d] * qkv[base + d];
            }
            let rms = (ss / head_dim as f32 + eps).sqrt().recip();
            for d in 0..head_dim {
                q_expected[mi * n_q + base + d] = qkv[base + d] * rms * gamma_qk[d];
            }
        }
        for h in 0..num_kv_heads {
            let base = n_q + h * head_dim;
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                ss += qkv[base + d] * qkv[base + d];
            }
            let rms = (ss / head_dim as f32 + eps).sqrt().recip();
            for d in 0..head_dim {
                k_expected[mi * n_k + h * head_dim + d] = qkv[base + d] * rms * gamma_qk[d];
            }
        }
        for d in 0..n_v {
            v_expected[mi * n_v + d] = qkv[n_q + n_k + d];
        }
    }
    cpu_rope_in_place(
        &mut q_expected,
        m,
        num_q_heads,
        head_dim,
        rotary_dim,
        theta,
        &positions,
    );
    cpu_rope_in_place(
        &mut k_expected,
        m,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
        &positions,
    );

    let mut w_packed = vec![0u8; (n_total * k) / 2];
    for j in 0..(n_total * k) {
        let code = w_codes[j] & 0x0F;
        if j % 2 == 0 {
            w_packed[j / 2] |= code;
        } else {
            w_packed[j / 2] |= code << 4;
        }
    }
    let num_blocks = (n_total * k) / 32;
    let w_exps: Vec<u8> = vec![shared_exp; num_blocks];

    let x_shape = Shape::from_slice(&[m, k]);
    let gamma_shape = Shape::from_slice(&[head_dim]);
    let w_shape = Shape::from_slice(&[n_total, k]);
    let exps_shape = Shape::from_slice(&[num_blocks]);
    let q_shape = Shape::from_slice(&[m, n_q]);
    let kv_shape = Shape::from_slice(&[max_seq_len, n_k]);
    let pos_shape = Shape::from_slice(&[m]);

    let x_dev = BackendDevice::from_cpu(&dev, &x_data, &x_shape, DType::F32)?;
    let gamma_dev = BackendDevice::from_cpu(&dev, &gamma_qk, &gamma_shape, DType::F32)?;
    let w_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_packed,
        &w_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        },
    )?;
    let exps_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_exps,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;
    let pos_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&positions), &pos_shape, DType::U32)?;

    let q_out_dev = BackendDevice::zeros(&dev, &q_shape, DType::F32)?;
    let k_cache_dev = BackendDevice::zeros(&dev, &kv_shape, DType::F32)?;
    let v_cache_dev = BackendDevice::zeros(&dev, &kv_shape, DType::F32)?;

    let handle = dev.fused_mxfp4_gemm_qk_norm_rope_kv(
        x_dev.as_ref(),
        gamma_dev.as_ref(),
        gamma_dev.as_ref(),
        w_dev.as_ref(),
        exps_dev.as_ref(),
        Some(q_out_dev.as_ref()),
        Some(k_cache_dev.as_ref()),
        Some(v_cache_dev.as_ref()),
        None,
        Some(pos_dev.as_ref()),
        m,
        k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
        None,
        1.0f32,
        eps,
        max_seq_len,
    )?;
    handle.synchronize()?;

    let q_actual = q_out_dev.to_cpu_vec_f32()?;
    let k_actual = k_cache_dev.to_cpu_vec_f32()?;
    let v_actual = v_cache_dev.to_cpu_vec_f32()?;

    for (i, (&a, &e)) in q_actual.iter().zip(q_expected.iter()).enumerate() {
        let err = (a - e).abs();
        assert!(
            err < 1e-3,
            "Q mismatch at {i}: actual={a}, expected={e}, err={err}"
        );
    }
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_k {
            let err = (k_actual[pos * n_k + c] - k_expected[row * n_k + c]).abs();
            assert!(err < 1e-3, "K mismatch pos {pos} col {c}");
        }
    }
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_v {
            let err = (v_actual[pos * n_v + c] - v_expected[row * n_v + c]).abs();
            assert!(err < 1e-3, "V mismatch pos {pos} col {c}");
        }
    }

    Ok(())
}

/// LFM2-style fused QKV with YaRN RoPE (inv_freq ramp + mscale).
// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn test_fused_mxfp4_gemm_qk_norm_rope_kv_yarn_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 64usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 16usize;
    let rotary_dim = 16usize;
    let theta = 10000.0f32;
    let eps = 1e-5f32;
    let max_seq_len = 16usize;

    let yarn_factor = 2.0f32;
    let original_max_pos = 32.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let attention_factor = 1.0f32;
    let mscale = attention_factor;

    let n_q = num_q_heads * head_dim;
    let n_k = num_kv_heads * head_dim;
    let n_v = num_kv_heads * head_dim;
    let n_total = n_q + n_k + n_v;

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.08).sin()).collect();
    let gamma_q: Vec<f32> = vec![1.0f32; head_dim];
    let gamma_k: Vec<f32> = (0..head_dim)
        .map(|d| 1.0f32 + 0.1f32 * (d as f32))
        .collect();
    let positions: Vec<u32> = vec![0, 1];

    let shared_exp = 127u8;
    let w_orig: Vec<f32> = (0..n_total * k)
        .map(|i| (i as f32 * 0.02).cos() * 1.2)
        .collect();
    let w_codes: Vec<u8> = w_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let w_dequant: Vec<f32> = w_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

    // CPU oracle: GEMM -> per-head QK-Norm -> YaRN RoPE
    let mut q_expected = vec![0.0f32; m * n_q];
    let mut k_expected = vec![0.0f32; m * n_k];
    let mut v_expected = vec![0.0f32; 0];
    v_expected = vec![0.0f32; m * n_v];
    for mi in 0..m {
        let mut qkv = vec![0.0f32; n_total];
        for ni in 0..n_total {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += x_data[mi * k + ki] * w_dequant[ni * k + ki];
            }
            qkv[ni] = sum;
        }
        for h in 0..num_q_heads {
            let base = h * head_dim;
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                ss += qkv[base + d] * qkv[base + d];
            }
            let rms = (ss / head_dim as f32 + eps).sqrt().recip();
            for d in 0..head_dim {
                q_expected[mi * n_q + base + d] = qkv[base + d] * rms * gamma_q[d];
            }
        }
        for h in 0..num_kv_heads {
            let base = n_q + h * head_dim;
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                ss += qkv[base + d] * qkv[base + d];
            }
            let rms = (ss / head_dim as f32 + eps).sqrt().recip();
            for d in 0..head_dim {
                k_expected[mi * n_k + h * head_dim + d] = qkv[base + d] * rms * gamma_k[d];
            }
        }
        for d in 0..n_v {
            v_expected[mi * n_v + d] = qkv[n_q + n_k + d];
        }
    }
    let rotary_half = rotary_dim / 2;
    let inv_freq = cpu_yarn_inv_freq(
        rotary_half,
        head_dim,
        theta,
        yarn_factor,
        original_max_pos,
        beta_fast,
        beta_slow,
    );
    cpu_rope_yarn_in_place(
        &mut q_expected,
        m,
        num_q_heads,
        head_dim,
        rotary_dim,
        &inv_freq,
        mscale,
        &positions,
    );
    cpu_rope_yarn_in_place(
        &mut k_expected,
        m,
        num_kv_heads,
        head_dim,
        rotary_dim,
        &inv_freq,
        mscale,
        &positions,
    );

    let mut w_packed = vec![0u8; (n_total * k) / 2];
    for j in 0..(n_total * k) {
        let code = w_codes[j] & 0x0F;
        if j % 2 == 0 {
            w_packed[j / 2] |= code;
        } else {
            w_packed[j / 2] |= code << 4;
        }
    }
    let num_blocks = (n_total * k) / 32;
    let w_exps: Vec<u8> = vec![shared_exp; num_blocks];

    let x_shape = Shape::from_slice(&[m, k]);
    let gamma_shape = Shape::from_slice(&[head_dim]);
    let w_shape = Shape::from_slice(&[n_total, k]);
    let exps_shape = Shape::from_slice(&[num_blocks]);
    let inv_freq_shape = Shape::from_slice(&[rotary_half]);
    let q_shape = Shape::from_slice(&[m, n_q]);
    let kv_shape = Shape::from_slice(&[max_seq_len, n_k]);
    let pos_shape = Shape::from_slice(&[m]);

    let x_dev = BackendDevice::from_cpu(&dev, &x_data, &x_shape, DType::F32)?;
    let gamma_q_dev = BackendDevice::from_cpu(&dev, &gamma_q, &gamma_shape, DType::F32)?;
    let gamma_k_dev = BackendDevice::from_cpu(&dev, &gamma_k, &gamma_shape, DType::F32)?;
    let w_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_packed,
        &w_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        },
    )?;
    let exps_dev = BackendDevice::from_cpu_bytes(
        &dev,
        &w_exps,
        &exps_shape,
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )?;
    let inv_freq_dev = BackendDevice::from_cpu(&dev, &inv_freq, &inv_freq_shape, DType::F32)?;
    let pos_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&positions), &pos_shape, DType::U32)?;

    let q_out_dev = BackendDevice::zeros(&dev, &q_shape, DType::F32)?;
    let k_cache_dev = BackendDevice::zeros(&dev, &kv_shape, DType::F32)?;
    let v_cache_dev = BackendDevice::zeros(&dev, &kv_shape, DType::F32)?;

    let handle = dev.fused_mxfp4_gemm_qk_norm_rope_kv(
        x_dev.as_ref(),
        gamma_q_dev.as_ref(),
        gamma_k_dev.as_ref(),
        w_dev.as_ref(),
        exps_dev.as_ref(),
        Some(q_out_dev.as_ref()),
        Some(k_cache_dev.as_ref()),
        Some(v_cache_dev.as_ref()),
        None,
        Some(pos_dev.as_ref()),
        m,
        k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
        Some(inv_freq_dev.as_ref()),
        mscale,
        eps,
        max_seq_len,
    )?;
    handle.synchronize()?;

    let q_actual = q_out_dev.to_cpu_vec_f32()?;
    let k_actual = k_cache_dev.to_cpu_vec_f32()?;
    let v_actual = v_cache_dev.to_cpu_vec_f32()?;

    for (i, (&a, &e)) in q_actual.iter().zip(q_expected.iter()).enumerate() {
        let err = (a - e).abs();
        assert!(
            err < 1e-3,
            "Q mismatch at {i}: actual={a}, expected={e}, err={err}"
        );
    }
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_k {
            let err = (k_actual[pos * n_k + c] - k_expected[row * n_k + c]).abs();
            assert!(err < 1e-3, "K mismatch pos {pos} col {c}");
        }
    }
    for row in 0..m {
        let pos = positions[row] as usize;
        for c in 0..n_v {
            let err = (v_actual[pos * n_v + c] - v_expected[row * n_v + c]).abs();
            assert!(err < 1e-3, "V mismatch pos {pos} col {c}");
        }
    }

    Ok(())
}
