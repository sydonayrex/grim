use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::QuantOps;
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage};
use grim_tensor::{CoreTensorOps, MemoryOps, QuantFormat, Shape};

fn main() {
    unsafe {
        std::env::set_var("HIP_VISIBLE_DEVICES", "0,1");
        std::env::set_var("GRIM_GPU_TEST", "1");
    }

    println!("=== STANDALONE SHAPE PARITY HARNESS ===");
    for &ord in &[0usize, 1usize] {
        let dev = RocmDevice::try_new(ord).expect("RocmDevice::try_new");
        println!("\n--- Testing on Device {} ---", ord);

        let test_cases = [
            (
                "iq4nl",
                QuantFormat::Iq4Nl,
                KQuantScheme::IQ4NL,
                17408,
                5120,
            ),
            (
                "iq4nl",
                QuantFormat::Iq4Nl,
                KQuantScheme::IQ4NL,
                5120,
                17408,
            ),
            (
                "iq4xs",
                QuantFormat::Iq4Xs,
                KQuantScheme::IQ4XS,
                17408,
                5120,
            ),
            (
                "iq4xs",
                QuantFormat::Iq4Xs,
                KQuantScheme::IQ4XS,
                5120,
                17408,
            ),
            ("iq3s", QuantFormat::Iq3S, KQuantScheme::IQ3S, 5120, 5120),
            ("iq3s", QuantFormat::Iq3S, KQuantScheme::IQ3S, 17408, 5120),
        ];

        for (tag, qfmt, scheme, n, k) in test_cases {
            let m = 64;
            // Generate synthetic activations with shape-dependent seed to avoid duplicate identical vectors
            let a_host: Vec<f32> = (0..m * k)
                .map(|i| (((i + n) % 100) as f32 - 50.0) * 0.01)
                .collect();
            let a_shape = Shape::new(vec![m, k]);
            let a_dev = dev
                .from_cpu(
                    &a_host,
                    &a_shape,
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )
                .expect("upload A");

            // Sizing for packed weights
            let block_bytes = match scheme {
                KQuantScheme::IQ4NL => 18,  // 32 elems -> 18 bytes
                KQuantScheme::IQ4XS => 136, // 256 elems -> 136 bytes
                KQuantScheme::IQ3S => 110,  // 256 elems -> 110 bytes
                _ => unreachable!(),
            };
            let block_size = match scheme {
                KQuantScheme::IQ4NL => 32,
                _ => 256,
            };
            let n_blocks = (n * k) / block_size;
            let weight_raw_bytes = n_blocks * block_bytes;

            // Generate deterministic non-zero synthetic packed weights with valid FP16 d scale
            let mut b_host_bytes: Vec<u8> = (0..weight_raw_bytes)
                .map(|i| ((i * 17 + 3) % 251) as u8)
                .collect();
            // Force d (first 2 bytes of every block) to a valid normal FP16 value, e.g. 0x3C00 (1.0 in FP16)
            for b in 0..n_blocks {
                let off = b * block_bytes;
                b_host_bytes[off] = 0x00;
                b_host_bytes[off + 1] = 0x3C; // 1.0f in FP16
            }

            let w_dev = dev
                .from_cpu_bytes(
                    &b_host_bytes,
                    &Shape::new(vec![n, k]),
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::KQuant(scheme),
                    },
                )
                .expect("upload packed weight");

            let out_shape = Shape::new(vec![m, n]);

            // Run scalar reference
            unsafe {
                std::env::set_var("GRIM_PREFILL_DEQUANT_GEMM", "0");
            }
            let (out_scalar, _) = dev
                .quantized_matmul(a_dev.as_ref(), w_dev.as_ref(), &[], qfmt, &out_shape)
                .expect("device scalar matmul");
            let res_scalar = out_scalar.to_cpu_vec_f32().expect("read scalar out");

            // Run dequant + rocBLAS
            unsafe {
                std::env::set_var("GRIM_PREFILL_DEQUANT_GEMM", "1");
            }
            let (out_dequant, _) = dev
                .quantized_matmul(a_dev.as_ref(), w_dev.as_ref(), &[], qfmt, &out_shape)
                .expect("device dequant matmul");
            let res_dequant = out_dequant.to_cpu_vec_f32().expect("read dequant out");

            // Compute max diff and relative error
            let mut max_abs = 0.0f32;
            let mut max_ref = 0.0f32;
            for i in 0..m * n {
                let diff = (res_scalar[i] - res_dequant[i]).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                let ref_val = res_scalar[i].abs();
                if ref_val > max_ref {
                    max_ref = ref_val;
                }
            }
            let rel_err = if max_ref > 0.0 {
                max_abs / max_ref
            } else {
                0.0
            };
            println!(
                "dev{} {:<6} [M={:<3}, N={:<5}, K={:<5}]: max_diff={:.6}, max_ref={:.2}, rel_err={:.2e} (sample scalar[0]={:.4}, dequant[0]={:.4})",
                ord, tag, m, n, k, max_abs, max_ref, rel_err, res_scalar[0], res_dequant[0]
            );
            assert!(
                rel_err < 1.0e-3,
                "Parity check failed for {} on dev{}",
                tag,
                ord
            );
        }
    }
    println!("\nALL SHAPE PARITY CHECKS PASSED!");
}
