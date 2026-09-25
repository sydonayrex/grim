//! native epilogue tests — split from the original monolithic lib_internal_tests.rs.

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn rocm_native_rope_device_gated_parity() {
        if !crate::gpu_test_enabled() {
            eprintln!("ROCm device tests disabled: skipping rocm_native_rope_device_gated_parity");
            return;
        }
        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("ROCm device unavailable: skipping rocm_native_rope_device_gated_parity");
                return;
            }
        };

        let dim = 4;
        let base = 10000.0_f32;
        let positions = [5u32];
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![1, 1, 4]);

        let in_storage = dev.from_cpu(&input, &shape, DType::F32).expect("from_cpu");
        let (out_storage, _handle) = dev
            .rope(
                in_storage.as_ref(),
                &positions,
                &grim_tensor::RopeConfig::new(dim, base),
                &shape,
            )
            .expect("dev.rope");
        let got = out_storage.to_cpu_vec_f32().expect("to_cpu_vec_f32");

        let inv_freq = [1.0_f32, 1.0 / 10000.0_f32.powf(2.0 / 4.0)];
        let pos = 5.0_f32;
        let cos_p = [(pos * inv_freq[0]).cos(), (pos * inv_freq[1]).cos()];
        let sin_p = [(pos * inv_freq[0]).sin(), (pos * inv_freq[1]).sin()];
        // Interleaved pairing (x[2i], x[2i+1]) - matches the CPU `Rope::forward` oracle.
        // The previous half-split expectation here enshrined the kernel bug that corrupted LFM2.5 ROCm generation.
        let want = [
            input[0] * cos_p[0] - input[1] * sin_p[0],
            input[1] * cos_p[0] + input[0] * sin_p[0],
            input[2] * cos_p[1] - input[3] * sin_p[1],
            input[3] * cos_p[1] + input[2] * sin_p[1],
        ];

        assert_eq!(got.len(), 4);
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "RoPE ROCm device parity mismatch at [{i}]: got {g:.8}, want {w:.8}",
            );
        }
    }

    #[test]
    fn rocm_native_broadcast_bias_device_gated_parity() {
        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "ROCm device unavailable: skipping rocm_native_broadcast_bias_device_gated_parity"
                );
                return;
            }
        };

        let bias = vec![0.1_f32, 0.2, 0.3, 0.4];
        let batch = 3;
        let out_dim = 4;
        let bias_shape = Shape::new(vec![4]);
        let out_shape = Shape::new(vec![batch, out_dim]);

        let bias_storage = dev
            .from_cpu(&bias, &bias_shape, DType::F32)
            .expect("from_cpu");
        let (out_storage, _handle) = dev
            .broadcast_bias(bias_storage.as_ref(), batch, out_dim, &out_shape)
            .expect("dev.broadcast_bias");
        let got = out_storage.to_cpu_vec_f32().expect("to_cpu_vec_f32");

        let mut want = Vec::with_capacity(batch * out_dim);
        for _ in 0..batch {
            want.extend_from_slice(&bias);
        }

        assert_eq!(got.len(), batch * out_dim);
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "broadcast_bias ROCm device parity mismatch at [{i}]: got {g:.8}, want {w:.8}",
            );
        }
    }

    #[test]
    fn rocm_scale_bias_epilogue_device_gated_parity() {
        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "ROCm device unavailable: skipping rocm_scale_bias_epilogue_device_gated_parity"
                );
                return;
            }
        };

        let batch = 4;
        let out_dim = 5;
        let mut seed = 0x5EED_F00D_u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / 65536.0) - 1000.0
        };

        let a_scale: Vec<f32> = (0..batch).map(|_| rng()).collect();
        let b_scale: Vec<f32> = (0..out_dim).map(|_| rng()).collect();
        let bias: Vec<f32> = (0..out_dim).map(|_| rng()).collect();
        let gemm_out: Vec<f32> = (0..batch * out_dim).map(|_| rng()).collect();

        // CPU reference mirroring the kernel's rounding order exactly (s = a_scale*b_scale rounded, then v = out*s rounded, then +
        // bias), so the parity check is bit-exact rather than tolerance-limited at large magnitudes where 1 ulp dwarfs any fixed tolerance.
        let mut want = gemm_out.clone();
        for (i, &a_s) in a_scale.iter().enumerate() {
            for (j, &b_s) in b_scale.iter().enumerate() {
                let idx = i * out_dim + j;
                let mut s = 1.0f32;
                s *= a_s;
                s *= b_s;
                let mut v = gemm_out[idx] * s;
                v += bias[j];
                want[idx] = v;
            }
        }

        let out_shape = Shape::new(vec![batch, out_dim]);
        let a_shape = Shape::new(vec![batch]);
        let b_shape = Shape::new(vec![out_dim]);
        let bias_shape = Shape::new(vec![out_dim]);

        let out_storage = dev
            .from_cpu(&gemm_out, &out_shape, DType::F32)
            .expect("from_cpu out");
        let a_storage = dev
            .from_cpu(&a_scale, &a_shape, DType::F32)
            .expect("from_cpu a");
        let b_storage = dev
            .from_cpu(&b_scale, &b_shape, DType::F32)
            .expect("from_cpu b");
        let bias_storage = dev
            .from_cpu(&bias, &bias_shape, DType::F32)
            .expect("from_cpu bias");

        let _handle = dev
            .scale_bias_epilogue(
                out_storage.as_ref(),
                Some(a_storage.as_ref()),
                Some(b_storage.as_ref()),
                Some(bias_storage.as_ref()),
                batch,
                out_dim,
            )
            .expect("dev.scale_bias_epilogue");

        let got = out_storage.to_cpu_vec_f32().expect("to_cpu_vec_f32");
        assert_eq!(got.len(), batch * out_dim);
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "scale_bias_epilogue ROCm device parity mismatch at [{i}]: got {g:.8}, want {w:.8}",
            );
        }
    }
}
