//! Integration and Numerical Parity Tests for Production ROCm Kernels:
//! FlashDecoding (Split-KV Attention), DeepSeek MLA Decode, Marlin W4A16, and BitNet b1.58.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, Shape, dtype::DType};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

fn as_u8_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[test]
fn test_production_kernels_compile_and_discovered() {
    let dev = RocmDevice::new(0);
    let src = grim_backend_rocm::kernels::source_asm::compute_kernel_source();
    assert!(src.contains("grim_flash_decode_stage1"));
    assert!(src.contains("grim_flash_decode_stage2"));
    assert!(src.contains("grim_mla_absorbed_decode"));
    assert!(src.contains("grim_marlin_gemm_w4a16"));
    assert!(src.contains("grim_bitnet_gemm_w158a8"));
    assert_eq!(dev.wavefront_size(), grim_backend_rocm::WavefrontSize::W32);
}

#[test]
fn test_flash_decode_split_kv_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let kv_seq_len = 512usize;
    let num_splits = 4usize;
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    let q_data: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let k_data: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();
    let v_data: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.04).sin())
        .collect();

    // CPU Golden Reference Flash Attention
    let mut expected_out = vec![0.0f32; num_heads * head_dim];
    let q_per_kv = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / q_per_kv;
        let mut running_max = -1e20f32;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];

        for j in 0..kv_seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                let q_val = q_data[h * head_dim + d];
                let k_val = k_data[(j * num_kv_heads + kv_h) * head_dim + d];
                dot += q_val * k_val;
            }
            let score = dot * inv_sqrt_d;

            let new_max = running_max.max(score);
            let alpha = (running_max - new_max).exp();
            let beta = (score - new_max).exp();

            running_sum = running_sum * alpha + beta;
            running_max = new_max;

            for d in 0..head_dim {
                let v_val = v_data[(j * num_kv_heads + kv_h) * head_dim + d];
                acc[d] = acc[d] * alpha + beta * v_val;
            }
        }

        for d in 0..head_dim {
            expected_out[h * head_dim + d] = acc[d] / running_sum;
        }
    }

    let q_shape = Shape::from_slice(&[1, num_heads, head_dim]);
    let kv_shape = Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]);
    let out_shape = Shape::from_slice(&[1, num_heads, head_dim]);

    let q_dev = BackendDevice::from_cpu(&dev, &q_data, &q_shape, DType::F32)?;
    let k_dev = BackendDevice::from_cpu(&dev, &k_data, &kv_shape, DType::F32)?;
    let v_dev = BackendDevice::from_cpu(&dev, &v_data, &kv_shape, DType::F32)?;
    let out_dev = BackendDevice::zeros(&dev, &out_shape, DType::F32)?;

    let q_s = grim_backend_rocm::device::util::as_rocm(q_dev.as_ref())?;
    let k_s = grim_backend_rocm::device::util::as_rocm(k_dev.as_ref())?;
    let v_s = grim_backend_rocm::device::util::as_rocm(v_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    dev.launch_flash_decode(
        q_s,
        k_s,
        v_s,
        out_s,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        num_splits,
    )?;
    dev.synchronize();

    let actual_out = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual_out.len(), expected_out.len());

    for (i, (&act, &exp)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "FlashDecode mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}

#[test]
fn test_mla_absorbed_decode_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let num_heads = 2usize;
    let kv_lora_rank = 32usize;
    let qk_rope_dim = 16usize;
    let v_dim = 32usize;
    let seq_len = 64usize;
    let inv_sqrt_d = 1.0f32 / ((kv_lora_rank + qk_rope_dim) as f32).sqrt();

    let q_nope_data: Vec<f32> = (0..num_heads * kv_lora_rank)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let q_pe_data: Vec<f32> = (0..num_heads * qk_rope_dim)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();
    let kv_comp_data: Vec<f32> = (0..seq_len * kv_lora_rank)
        .map(|i| (i as f32 * 0.02).sin())
        .collect();
    let k_pe_data: Vec<f32> = (0..seq_len * qk_rope_dim)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    // Golden Reference MLA
    let mut expected_out = vec![0.0f32; num_heads * v_dim];

    for h in 0..num_heads {
        let mut running_max = -1e20f32;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; v_dim];

        for j in 0..seq_len {
            let mut dot_nope = 0.0f32;
            for d in 0..kv_lora_rank {
                dot_nope += q_nope_data[h * kv_lora_rank + d] * kv_comp_data[j * kv_lora_rank + d];
            }
            let mut dot_pe = 0.0f32;
            for d in 0..qk_rope_dim {
                dot_pe += q_pe_data[h * qk_rope_dim + d] * k_pe_data[j * qk_rope_dim + d];
            }
            let score = (dot_nope + dot_pe) * inv_sqrt_d;

            let new_max = running_max.max(score);
            let alpha = (running_max - new_max).exp();
            let beta = (score - new_max).exp();

            running_sum = running_sum * alpha + beta;
            running_max = new_max;

            for d in 0..v_dim {
                let v_val = kv_comp_data[j * kv_lora_rank + d];
                acc[d] = acc[d] * alpha + beta * v_val;
            }
        }

        for d in 0..v_dim {
            expected_out[h * v_dim + d] = acc[d] / running_sum;
        }
    }

    let mut packed_kv_data = Vec::with_capacity(seq_len * (kv_lora_rank + qk_rope_dim));
    for j in 0..seq_len {
        packed_kv_data.extend_from_slice(&kv_comp_data[j * kv_lora_rank..(j + 1) * kv_lora_rank]);
        packed_kv_data.extend_from_slice(&k_pe_data[j * qk_rope_dim..(j + 1) * qk_rope_dim]);
    }

    let q_nope_shape = Shape::from_slice(&[1, num_heads, kv_lora_rank]);
    let q_pe_shape = Shape::from_slice(&[1, num_heads, qk_rope_dim]);
    let packed_kv_shape = Shape::from_slice(&[seq_len, kv_lora_rank + qk_rope_dim]);
    let out_shape = Shape::from_slice(&[1, num_heads, v_dim]);

    let q_nope_dev = BackendDevice::from_cpu(&dev, &q_nope_data, &q_nope_shape, DType::F32)?;
    let q_pe_dev = BackendDevice::from_cpu(&dev, &q_pe_data, &q_pe_shape, DType::F32)?;
    let packed_kv_dev =
        BackendDevice::from_cpu(&dev, &packed_kv_data, &packed_kv_shape, DType::F32)?;
    let out_dev = BackendDevice::zeros(&dev, &out_shape, DType::F32)?;

    let q_nope_s = grim_backend_rocm::device::util::as_rocm(q_nope_dev.as_ref())?;
    let q_pe_s = grim_backend_rocm::device::util::as_rocm(q_pe_dev.as_ref())?;
    let packed_kv_s = grim_backend_rocm::device::util::as_rocm(packed_kv_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    dev.launch_mla_absorbed_decode(
        q_nope_s,
        q_pe_s,
        packed_kv_s,
        None,
        out_s,
        num_heads,
        kv_lora_rank,
        qk_rope_dim,
        v_dim,
        seq_len,
    )?;
    dev.synchronize();

    let actual_out = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual_out.len(), expected_out.len());

    for (i, (&act, &exp)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-3,
            "MLA Absorbed Decode mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}

#[test]
fn test_marlin_gemm_w4a16_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let n = 16usize;
    let k = 64usize;

    let mut a_f16: Vec<half::f16> = Vec::with_capacity(m * k);
    for i in 0..m * k {
        a_f16.push(half::f16::from_f32((i as f32 * 0.05).sin()));
    }

    let mut b_w4: Vec<u32> = vec![0x11111111; (k * n) / 8];
    for (i, v) in b_w4.iter_mut().enumerate() {
        *v = 0x22222222 ^ (i as u32);
    }
    let scales_len = n * (k / 16);
    let scales_f16: Vec<half::f16> = (0..scales_len).map(|_| half::f16::from_f32(0.5)).collect();

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[(k * n) / 8]);
    let scales_shape = Shape::from_slice(&[n, k / 16]);
    let out_shape = Shape::from_slice(&[m, n]);

    let a_dev = BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&a_f16), &a_shape, DType::F16)?;
    let b_dev = BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&b_w4), &b_shape, DType::U32)?;
    let scales_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&scales_f16), &scales_shape, DType::F16)?;
    let out_dev = BackendDevice::zeros(&dev, &out_shape, DType::F16)?;

    let a_s = grim_backend_rocm::device::util::as_rocm(a_dev.as_ref())?;
    let b_s = grim_backend_rocm::device::util::as_rocm(b_dev.as_ref())?;
    let scales_s = grim_backend_rocm::device::util::as_rocm(scales_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    dev.launch_marlin_gemm_w4a16(a_s, b_s, scales_s, out_s, m, n, k, 16)?;
    dev.synchronize();

    let actual_out = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual_out.len(), m * n);
    for val in actual_out {
        assert!(!val.is_nan() && !val.is_infinite());
    }

    Ok(())
}

#[test]
fn test_bitnet_gemm_w158a8_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let n = 16usize;
    let k = 32usize;

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) - 2.0).collect();
    let b_ternary_packed: Vec<u8> = vec![0b10010010; (n * k) / 4];
    let scales_b: Vec<f32> = (0..n).map(|i| 0.1 * (1.0 + i as f32)).collect();
    let scale_a = 0.5f32;

    // CPU Reference
    let mut expected_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum_val = 0.0f32;
            for ki in 0..k {
                let a_val = a_data[mi * k + ki];
                let elem_idx = ni * k + ki;
                let byte_idx = elem_idx / 4;
                let shift = (elem_idx % 4) * 2;
                let code = (b_ternary_packed[byte_idx] >> shift) & 0x3;
                let w_val = match code {
                    1 => 1.0f32,
                    2 => -1.0f32,
                    _ => 0.0f32,
                };
                sum_val += a_val * w_val;
            }
            expected_c[mi * n + ni] = sum_val * scale_a * scales_b[ni];
        }
    }

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[(n * k) / 4]);
    let scales_shape = Shape::from_slice(&[n]);
    let out_shape = Shape::from_slice(&[m, n]);

    let a_dev = BackendDevice::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
    let b_dev = BackendDevice::from_cpu_bytes(&dev, &b_ternary_packed, &b_shape, DType::U8)?;
    let scales_dev = BackendDevice::from_cpu(&dev, &scales_b, &scales_shape, DType::F32)?;
    let out_dev = BackendDevice::zeros(&dev, &out_shape, DType::F32)?;

    let a_s = grim_backend_rocm::device::util::as_rocm(a_dev.as_ref())?;
    let b_s = grim_backend_rocm::device::util::as_rocm(b_dev.as_ref())?;
    let scales_s = grim_backend_rocm::device::util::as_rocm(scales_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    dev.launch_bitnet_gemm_w158a8(a_s, b_s, scales_s, out_s, m, n, k, scale_a)?;
    dev.synchronize();

    let actual_c = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual_c.len(), expected_c.len());

    for (i, (&act, &exp)) in actual_c.iter().zip(expected_c.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "BitNet b1.58 GEMM mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}
