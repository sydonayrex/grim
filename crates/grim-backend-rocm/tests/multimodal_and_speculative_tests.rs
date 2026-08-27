//! Integration and Numerical Parity Tests for Multimodal 3D M-RoPE,
//! GPU Speculative Rejection Sampling, and EPLB Load Balancing.

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::eplb::EplbRouter;
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
fn test_multimodal_and_speculative_kernels_compile() {
    let dev = RocmDevice::new(0);
    let src = grim_backend_rocm::kernels::source_asm::compute_kernel_source();
    assert!(src.contains("grim_mrope_qk"));
    assert!(src.contains("grim_speculative_rejection_sample"));
    assert_eq!(dev.wavefront_size(), grim_backend_rocm::WavefrontSize::W32);
}

#[test]
fn test_eplb_multi_rank_load_balancing() {
    // 16 experts with severe skewed routing load
    let frequencies: Vec<f32> = vec![
        250.0, 220.0, 180.0, 150.0, 120.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0,
        15.0, 10.0,
    ];
    let num_ranks = 4;
    let replication_slots = 3;

    let plan = EplbRouter::balance_experts(&frequencies, num_ranks, replication_slots);
    assert_eq!(plan.expert_to_rank.len(), 16);
    assert_eq!(plan.rank_loads.len(), 4);

    let total_freq: f32 = frequencies.iter().sum();
    let total_packed: f32 = plan.rank_loads.iter().sum();
    assert!((total_freq - total_packed).abs() < 1e-4);

    // Peak-to-mean imbalance ratio must be tightly bounded (< 1.25)
    assert!(
        plan.imbalance_ratio() < 1.25,
        "Imbalance ratio was too high: {}",
        plan.imbalance_ratio()
    );

    // Exactly 3 hot experts replicated
    assert_eq!(plan.replicated_experts.len(), 3);
    assert_eq!(plan.replicated_experts[0].0, 0); // Expert 0 (load 250)
    assert_eq!(plan.replicated_experts[1].0, 1); // Expert 1 (load 220)
    assert_eq!(plan.replicated_experts[2].0, 2); // Expert 2 (load 180)
}

#[test]
fn test_mrope_numerical_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let num_tokens = 2usize;
    let num_q_heads = 4usize;
    let num_k_heads = 2usize;
    let head_dim = 64usize;
    let rotary_dim = 64usize;
    let (section_t, section_h, section_w) = (8usize, 12usize, 12usize); // Sum = 32 pairs = 64 dims
    let rope_theta = 10000.0f32;

    // 3D coordinates: Token 0 = (T=5, H=10, W=20), Token 1 = (T=6, H=11, W=21)
    let positions: Vec<i32> = vec![5, 10, 20, 6, 11, 21];

    let q_data: Vec<f32> = (0..num_tokens * num_q_heads * head_dim)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let k_data: Vec<f32> = (0..num_tokens * num_k_heads * head_dim)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();

    // CPU Golden Reference M-RoPE
    let mut expected_q = q_data.clone();
    let mut expected_k = k_data.clone();

    for t in 0..num_tokens {
        let pos_t = positions[t * 3];
        let pos_h = positions[t * 3 + 1];
        let pos_w = positions[t * 3 + 2];

        // Q heads
        for h in 0..num_q_heads {
            let base = (t * num_q_heads + h) * head_dim;
            for pair_idx in 0..(rotary_dim / 2) {
                let pos = if pair_idx < section_t {
                    pos_t
                } else if pair_idx < section_t + section_h {
                    pos_h
                } else {
                    pos_w
                };

                let freq = 1.0f32 / rope_theta.powf((2.0f32 * pair_idx as f32) / rotary_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let x0 = expected_q[base + 2 * pair_idx];
                let x1 = expected_q[base + 2 * pair_idx + 1];
                expected_q[base + 2 * pair_idx] = x0 * cos_val - x1 * sin_val;
                expected_q[base + 2 * pair_idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }

        // K heads
        for h in 0..num_k_heads {
            let base = (t * num_k_heads + h) * head_dim;
            for pair_idx in 0..(rotary_dim / 2) {
                let pos = if pair_idx < section_t {
                    pos_t
                } else if pair_idx < section_t + section_h {
                    pos_h
                } else {
                    pos_w
                };

                let freq = 1.0f32 / rope_theta.powf((2.0f32 * pair_idx as f32) / rotary_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let x0 = expected_k[base + 2 * pair_idx];
                let x1 = expected_k[base + 2 * pair_idx + 1];
                expected_k[base + 2 * pair_idx] = x0 * cos_val - x1 * sin_val;
                expected_k[base + 2 * pair_idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    let q_shape = Shape::from_slice(&[num_tokens, num_q_heads, head_dim]);
    let k_shape = Shape::from_slice(&[num_tokens, num_k_heads, head_dim]);
    let pos_shape = Shape::from_slice(&[num_tokens, 3]);

    let q_dev = BackendDevice::from_cpu(&dev, &q_data, &q_shape, DType::F32)?;
    let k_dev = BackendDevice::from_cpu(&dev, &k_data, &k_shape, DType::F32)?;
    let pos_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&positions), &pos_shape, DType::U32)?;

    let q_s = grim_backend_rocm::device::util::as_rocm(q_dev.as_ref())?;
    let k_s = grim_backend_rocm::device::util::as_rocm(k_dev.as_ref())?;
    let pos_s = grim_backend_rocm::device::util::as_rocm(pos_dev.as_ref())?;

    dev.launch_mrope_qk(
        q_s,
        k_s,
        pos_s,
        num_tokens,
        num_q_heads,
        num_k_heads,
        head_dim,
        rotary_dim,
        section_t,
        section_h,
        section_w,
        rope_theta,
    )?;
    dev.synchronize();

    let actual_q = q_dev.to_cpu_vec_f32()?;
    let actual_k = k_dev.to_cpu_vec_f32()?;

    for (i, (&act, &exp)) in actual_q.iter().zip(expected_q.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "M-RoPE Q mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }
    for (i, (&act, &exp)) in actual_k.iter().zip(expected_k.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "M-RoPE K mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}

#[test]
fn test_speculative_rejection_sampler_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let batch_size = 1usize;
    let num_draft_tokens = 3usize;
    let vocab_size = 4usize;

    // Target probs: [batch_size, num_draft_tokens + 1, vocab_size]
    let target_probs = vec![
        // Token 0: high prob on token 1
        0.1f32, 0.8f32, 0.05f32, 0.05f32, // Token 1: high prob on token 2
        0.05f32, 0.05f32, 0.8f32, 0.1f32,
        // Token 2: high prob on token 0 (draft has token 3 -> will reject)
        0.85f32, 0.05f32, 0.05f32, 0.05f32, // Bonus Token 3:
        0.1f32, 0.1f32, 0.1f32, 0.7f32,
    ];

    // Draft probs: [batch_size, num_draft_tokens, vocab_size]
    let draft_probs = vec![
        // Draft 0: predicts token 1 with prob 0.8
        0.1f32, 0.8f32, 0.05f32, 0.05f32, // Draft 1: predicts token 2 with prob 0.75
        0.05f32, 0.1f32, 0.75f32, 0.1f32,
        // Draft 2: incorrectly predicts token 3 with prob 0.9
        0.02f32, 0.03f32, 0.05f32, 0.9f32,
    ];

    let draft_tokens = vec![1i32, 2i32, 3i32];
    // Random numbers: token 0 (0.1 <= 1.0 -> accept), token 1 (0.2 <= 0.8/0.75 -> accept), token 2 (0.9 > 0.05/0.9 -> reject)
    // Random number for residual sampling: 0.1
    let uniform_rands = vec![0.1f32, 0.2f32, 0.95f32, 0.1f32];

    let tp_shape = Shape::from_slice(&[batch_size, num_draft_tokens + 1, vocab_size]);
    let dp_shape = Shape::from_slice(&[batch_size, num_draft_tokens, vocab_size]);
    let dt_shape = Shape::from_slice(&[batch_size, num_draft_tokens]);
    let ur_shape = Shape::from_slice(&[batch_size, num_draft_tokens + 1]);
    let out_shape = Shape::from_slice(&[batch_size, num_draft_tokens + 1]);
    let len_shape = Shape::from_slice(&[batch_size]);

    let tp_dev = BackendDevice::from_cpu(&dev, &target_probs, &tp_shape, DType::F32)?;
    let dp_dev = BackendDevice::from_cpu(&dev, &draft_probs, &dp_shape, DType::F32)?;
    let dt_dev =
        BackendDevice::from_cpu_bytes(&dev, as_u8_slice(&draft_tokens), &dt_shape, DType::U32)?;
    let ur_dev = BackendDevice::from_cpu(&dev, &uniform_rands, &ur_shape, DType::F32)?;
    let out_dev = BackendDevice::zeros(&dev, &out_shape, DType::U32)?;
    let len_dev = BackendDevice::zeros(&dev, &len_shape, DType::U32)?;

    let tp_s = grim_backend_rocm::device::util::as_rocm(tp_dev.as_ref())?;
    let dp_s = grim_backend_rocm::device::util::as_rocm(dp_dev.as_ref())?;
    let dt_s = grim_backend_rocm::device::util::as_rocm(dt_dev.as_ref())?;
    let ur_s = grim_backend_rocm::device::util::as_rocm(ur_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;
    let len_s = grim_backend_rocm::device::util::as_rocm(len_dev.as_ref())?;

    dev.launch_speculative_rejection_sample(
        tp_s,
        dp_s,
        dt_s,
        ur_s,
        out_s,
        len_s,
        batch_size,
        num_draft_tokens,
        vocab_size,
    )?;
    dev.synchronize();

    let mut out_tokens = vec![0i32; num_draft_tokens + 1];
    let mut out_lens = vec![0i32; 1];

    let bytes = out_dev
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();
    let mut host_tokens = vec![0u8; out_tokens.len() * 4];
    grim_backend_rocm::check_hip("DtoH tokens", unsafe {
        grim_backend_rocm::hipMemcpy(
            host_tokens.as_mut_ptr() as *mut std::ffi::c_void,
            bytes.device_ptr_u64().unwrap() as *const std::ffi::c_void,
            host_tokens.len(),
            grim_backend_rocm::HipMemcpyKind::DeviceToHost,
        )
    })?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_tokens.as_ptr() as *const i32,
            out_tokens.as_mut_ptr(),
            out_tokens.len(),
        );
    }

    let len_s = len_dev
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();
    let mut host_lens = vec![0u8; 4];
    grim_backend_rocm::check_hip("DtoH len", unsafe {
        grim_backend_rocm::hipMemcpy(
            host_lens.as_mut_ptr() as *mut std::ffi::c_void,
            len_s.device_ptr_u64().unwrap() as *const std::ffi::c_void,
            4,
            grim_backend_rocm::HipMemcpyKind::DeviceToHost,
        )
    })?;
    unsafe {
        std::ptr::copy_nonoverlapping(host_lens.as_ptr() as *const i32, out_lens.as_mut_ptr(), 1);
    }

    // Token 0 (1) accepted, Token 1 (2) accepted, Token 2 (3) rejected -> sampled token 0 from residual
    assert_eq!(out_lens[0], 3); // 2 accepted + 1 recovery = 3
    assert_eq!(out_tokens[0], 1);
    assert_eq!(out_tokens[1], 2);
    assert_eq!(out_tokens[2], 0); // Residual sampled token 0

    Ok(())
}
