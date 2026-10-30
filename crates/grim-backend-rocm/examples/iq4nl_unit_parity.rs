use grim_backend_rocm::{RocmDevice, as_rocm};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

fn main() {
    unsafe {
        std::env::set_var("HIP_VISIBLE_DEVICES", "0,1");
        std::env::set_var("GRIM_GPU_TEST", "1");
    }

    println!("=== IQ4_NL UNIT PARITY HARNESS (m=1 decode) ===");
    for &ord in &[0usize, 1usize] {
        let dev = RocmDevice::try_new(ord).expect("RocmDevice::try_new");
        println!("\n--- Testing on Device {} ---", ord);

        // Test production shapes: N=5120, K=17408 and N=17408, K=5120
        let shapes = [(5120, 17408), (17408, 5120)];

        for (n, k) in shapes {
            let m = 1;
            // Generate synthetic activations
            let a_host: Vec<f32> = (0..k).map(|i| ((i % 100) as f32 - 50.0) * 0.02).collect();
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

            // Synthetic IQ4_NL packed weights
            // 32 elems -> 18 bytes (2 bytes f16 d + 16 bytes qs)
            let blocks_per_col = k / 32;
            let col_bytes = blocks_per_col * 18;
            let total_bytes = col_bytes * n;
            let mut w_bytes = vec![0u8; total_bytes];

            // Fill with deterministic patterns
            for col in 0..n {
                for blk in 0..blocks_per_col {
                    let offset = col * col_bytes + blk * 18;
                    // d = 0.5 in fp16 (0x3800)
                    w_bytes[offset] = 0x00;
                    w_bytes[offset + 1] = 0x38;
                    // nibbles: 0..15
                    for j in 0..16 {
                        let lo = ((col + blk + j) % 16) as u8;
                        let hi = ((col + blk + j + 8) % 16) as u8;
                        w_bytes[offset + 2 + j] = lo | (hi << 4);
                    }
                }
            }

            let w_shape = Shape::new(vec![n, k]);
            let w_dev = dev
                .from_cpu_bytes(
                    &w_bytes,
                    &w_shape,
                    DType {
                        arith: ArithType::U8,
                        storage: Storage::KQuant(KQuantScheme::IQ4NL),
                    },
                )
                .expect("upload W");

            let a_r = as_rocm(a_dev.as_ref()).unwrap();
            let w_r = as_rocm(w_dev.as_ref()).unwrap();

            // Prepare outputs
            let out_shape = Shape::new(vec![m, n]);
            let out_ref = dev
                .zeros(
                    &out_shape,
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )
                .expect("alloc out_ref");
            let out_new = dev
                .zeros(
                    &out_shape,
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )
                .expect("alloc out_new");

            let o_ref = as_rocm(out_ref.as_ref()).unwrap();
            let o_new = as_rocm(out_new.as_ref()).unwrap();

            // 1. Launch reference scalar kernel
            dev.launch_fused_dequant_gemm_iq4nl_for_ab(a_r, w_r, o_ref, m, n, k)
                .expect("launch scalar");

            // 2. Launch new dot4 kernel via Q8_1 act quant
            let q81_bytes = (k / 32) * 36 * m;
            let act_q81 = dev
                .zeros(
                    &Shape::new(vec![q81_bytes]),
                    DType {
                        arith: ArithType::U8,
                        storage: Storage::Native,
                    },
                )
                .expect("alloc act_q81");
            let act_q81_r = as_rocm(act_q81.as_ref()).unwrap();
            dev.launch_quantize_q8_1(a_r, act_q81_r, m, k)
                .expect("launch quantize_q8_1");
            dev.launch_dot4_iq4nl_q81_gemv_for_ab(act_q81_r, w_r, o_new, m, n, k)
                .expect("launch dot4");

            dev.synchronize();

            let ref_host: Vec<f32> = out_ref.to_cpu_vec_f32().expect("readback ref");
            let new_host: Vec<f32> = out_new.to_cpu_vec_f32().expect("readback new");

            let mut max_abs = 0.0f32;
            let mut sum_sq_diff = 0.0f64;
            let mut sum_sq_ref = 0.0f64;

            for i in 0..ref_host.len() {
                let r = ref_host[i];
                let n_val = new_host[i];
                let diff = (r - n_val).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                sum_sq_diff += (diff as f64) * (diff as f64);
                sum_sq_ref += (r as f64) * (r as f64);
            }

            let max_ref = ref_host.iter().fold(0f32, |x, v| x.max(v.abs())).max(1e-6);
            let worst_rel_to_max = max_abs / max_ref;
            let rms_rel = (sum_sq_diff / sum_sq_ref.max(1e-12)).sqrt();

            println!(
                "Shape N={} K={}: max_abs={:.6}, max_ref={:.2}, worst_rel={:.4e}, rms_rel={:.4e}",
                n, k, max_abs, max_ref, worst_rel_to_max, rms_rel
            );
            assert!(
                rms_rel < 0.01,
                "IQ4_NL dot4 rms_rel ({}) exceeds 1.0% threshold!",
                rms_rel
            );
        }
    }
    println!("\nALL IQ4_NL UNIT PARITY CHECKS PASSED!");
}
