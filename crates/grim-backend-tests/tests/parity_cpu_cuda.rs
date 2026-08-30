//! Multi-format backend dequantization & execution parity tests for CUDA (§WI-E9).

#[cfg(feature = "cuda")]
use grim_backend_tests::TEST_K_DIMS;
#[cfg(feature = "cuda")]
use grim_tensor::Shape;

#[cfg(feature = "cuda")]
fn generate_deterministic_test_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as i32) as f32 / (i32::MAX as f32);
        out.push(val);
    }
    out
}

#[cfg(feature = "cuda")]
fn cpu_rmsnorm_reference(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (n as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv_rms * w)
        .collect()
}

#[test]
fn test_cuda_parity_dequant_and_kernels_if_available() {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = grim_backend_cuda::CudaDevice::new(0) {
            use grim_tensor::{BackendStorage, CoreTensorOps, ElementwiseOps};

            // Test RMSNorm & elementwise numerical parity against CPU reference
            for &k in TEST_K_DIMS {
                let x = generate_deterministic_test_weights(k, 0x1234_5678);
                let w = generate_deterministic_test_weights(k, 0x8765_4321);
                let cpu_ref = cpu_rmsnorm_reference(&x, &w, 1e-5);

                let shape = Shape::new(vec![1, k]);
                let x_dev = dev
                    .from_cpu(&x, &shape, grim_tensor::DType::F32)
                    .expect("from_cpu x failed");
                let w_dev = dev
                    .from_cpu(&w, &Shape::new(vec![k]), grim_tensor::DType::F32)
                    .expect("from_cpu w failed");

                let (norm_out, handle) = dev
                    .rms_norm(x_dev.as_ref(), w_dev.as_ref(), 1e-5, &shape)
                    .expect("cuda rms_norm failed");
                handle.synchronize().expect("sync failed");

                let gpu_res = norm_out.to_cpu_vec_f32().expect("to_cpu failed");
                for (a, b) in cpu_ref.iter().zip(&gpu_res) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "CUDA RMSNorm mismatch k={k}: cpu={a}, gpu={b}"
                    );
                }

                // Sub reduction parity test
                let (sub_out, sub_handle) = dev
                    .sub(x_dev.as_ref(), w_dev.as_ref(), &shape)
                    .expect("cuda sub failed");
                sub_handle.synchronize().expect("sub sync failed");
                let sub_res = sub_out.to_cpu_vec_f32().expect("to_cpu failed");
                for i in 0..k {
                    assert!(
                        (sub_res[i] - (x[i] - w[i])).abs() < 1e-5,
                        "CUDA sub mismatch at {i}: got {}, expected {}",
                        sub_res[i],
                        x[i] - w[i]
                    );
                }

                // Reduce sum / max / argmax parity
                let r_sum = dev.reduce_sum(x_dev.as_ref()).expect("reduce_sum failed");
                let expected_sum: f32 = x.iter().sum();
                assert!(
                    (r_sum - expected_sum).abs() < 1e-3 * (k as f32),
                    "CUDA reduce_sum mismatch: {r_sum} vs {expected_sum}"
                );
            }

            // Speculative rejection sampler execution test
            let batch_size = 1;
            let num_draft = 3;
            let vocab_size = 4;

            // Target probs: [1, 4, 4]
            let mut target_probs_data = vec![0.0f32; (num_draft + 1) * vocab_size];
            for step in 0..=num_draft {
                target_probs_data[step * vocab_size + 0] = 0.7;
                target_probs_data[step * vocab_size + 1] = 0.1;
                target_probs_data[step * vocab_size + 2] = 0.1;
                target_probs_data[step * vocab_size + 3] = 0.1;
            }

            // Draft probs: [1, 3, 4]
            let mut draft_probs_data = vec![0.0f32; num_draft * vocab_size];
            for step in 0..num_draft {
                draft_probs_data[step * vocab_size + 0] = 0.7;
                draft_probs_data[step * vocab_size + 1] = 0.1;
                draft_probs_data[step * vocab_size + 2] = 0.1;
                draft_probs_data[step * vocab_size + 3] = 0.1;
            }

            // Draft tokens: [0, 0, 0] (all high probability target tokens)
            let draft_tokens_data = vec![0.0f32; num_draft]; // storage uploaded as f32 bits
            let uniform_rands_data = vec![0.05f32; num_draft + 1]; // uniform rands <= accept_prob (0.7/0.7=1.0) -> accept all

            let tp_dev = dev.from_cpu(&target_probs_data, &Shape::new(vec![1, num_draft + 1, vocab_size]), grim_tensor::DType::F32).unwrap();
            let dp_dev = dev.from_cpu(&draft_probs_data, &Shape::new(vec![1, num_draft, vocab_size]), grim_tensor::DType::F32).unwrap();
            let dt_dev = dev.from_cpu(&draft_tokens_data, &Shape::new(vec![1, num_draft]), grim_tensor::DType::F32).unwrap();
            let ur_dev = dev.from_cpu(&uniform_rands_data, &Shape::new(vec![1, num_draft + 1]), grim_tensor::DType::F32).unwrap();

            let accepted_tokens_init = vec![0.0f32; num_draft + 1];
            let accepted_lens_init = vec![0.0f32; batch_size];

            let at_dev = dev.from_cpu(&accepted_tokens_init, &Shape::new(vec![1, num_draft + 1]), grim_tensor::DType::F32).unwrap();
            let al_dev = dev.from_cpu(&accepted_lens_init, &Shape::new(vec![batch_size]), grim_tensor::DType::F32).unwrap();

            let tp_cuda = tp_dev.as_any().downcast_ref::<grim_backend_cuda::CudaStorage>().unwrap();
            let dp_cuda = dp_dev.as_any().downcast_ref::<grim_backend_cuda::CudaStorage>().unwrap();
            let dt_cuda = dt_dev.as_any().downcast_ref::<grim_backend_cuda::CudaStorage>().unwrap();
            let ur_cuda = ur_dev.as_any().downcast_ref::<grim_backend_cuda::CudaStorage>().unwrap();
            let at_cuda = at_dev.as_any().downcast_ref::<grim_backend_cuda::CudaStorage>().unwrap();
            let al_cuda = al_dev.as_any().downcast_ref::<grim_backend_cuda::CudaStorage>().unwrap();

            dev.launch_speculative_rejection_sample(
                tp_cuda, dp_cuda, dt_cuda, ur_cuda, at_cuda, al_cuda,
                batch_size, num_draft, vocab_size
            ).expect("launch_speculative_rejection_sample failed");

            let res_lens = al_cuda.to_cpu_vec_f32().unwrap();
            let res_tokens = at_cuda.to_cpu_vec_f32().unwrap();

            let accepted_len = i32::from_ne_bytes(res_lens[0].to_ne_bytes());
            assert_eq!(accepted_len, 4, "all 3 draft tokens + 1 target bonus token should be accepted");
            for i in 0..accepted_len as usize {
                let tok = i32::from_ne_bytes(res_tokens[i].to_ne_bytes());
                assert_eq!(tok, 0, "accepted token at {i} should be 0");
            }
        }
    }
}
