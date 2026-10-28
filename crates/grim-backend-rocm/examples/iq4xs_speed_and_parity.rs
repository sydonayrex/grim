use grim_backend_rocm::{RocmDevice, as_rocm};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

fn main() {
    unsafe {
        std::env::set_var("HIP_VISIBLE_DEVICES", "0,1");
        std::env::set_var("GRIM_GPU_TEST", "1");
    }

    println!("=== IQ4_XS SPEED & PARITY PROBE (m=1 decode) ===");
    for &ord in &[0usize, 1usize] {
        let dev = RocmDevice::try_new(ord).expect("RocmDevice::try_new");
        println!("\n--- Testing on Device {} ---", ord);

        let shapes = [(17408, 5120), (5120, 17408)];

        for (n, k) in shapes {
            let m = 1;
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

            let n_sb = k / 256;
            let row_bytes = n_sb * 136;
            let total_bytes = row_bytes * n;
            let mut w_bytes = vec![0u8; total_bytes];

            // Deterministic IQ4_XS pattern
            for col in 0..n {
                for sb in 0..n_sb {
                    let offset = col * row_bytes + sb * 136;
                    // d = 0.5 in fp16
                    w_bytes[offset] = 0x00;
                    w_bytes[offset + 1] = 0x38;
                    // scales_h = 0
                    w_bytes[offset + 2] = 0x00;
                    w_bytes[offset + 3] = 0x00;
                    // scales_l = 0x22 (subblock scale ls = 34 -> dl = d * 2)
                    w_bytes[offset + 4] = 0x22;
                    w_bytes[offset + 5] = 0x22;
                    w_bytes[offset + 6] = 0x22;
                    w_bytes[offset + 7] = 0x22;
                    // qs
                    for j in 0..128 {
                        let lo = ((col + sb + j) % 16) as u8;
                        let hi = ((col + sb + j + 8) % 16) as u8;
                        w_bytes[offset + 8 + j] = lo | (hi << 4);
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
                        storage: Storage::KQuant(KQuantScheme::IQ4XS),
                    },
                )
                .expect("upload W");

            let a_r = as_rocm(a_dev.as_ref()).unwrap();
            let w_r = as_rocm(w_dev.as_ref()).unwrap();

            let out_ref = dev
                .zeros(
                    &Shape::new(vec![m, n]),
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )
                .unwrap();
            let out_dot = dev
                .zeros(
                    &Shape::new(vec![m, n]),
                    DType {
                        arith: ArithType::F32,
                        storage: Storage::Native,
                    },
                )
                .unwrap();

            let o_ref = as_rocm(out_ref.as_ref()).unwrap();
            let o_dot = as_rocm(out_dot.as_ref()).unwrap();

            // Q8_1 act quant
            let q81_bytes = (k / 32) * 36 * m;
            let act_q81 = dev
                .zeros(
                    &Shape::new(vec![q81_bytes]),
                    DType {
                        arith: ArithType::U8,
                        storage: Storage::Native,
                    },
                )
                .unwrap();
            let act_q81_r = as_rocm(act_q81.as_ref()).unwrap();
            dev.launch_quantize_q8_1(a_r, act_q81_r, m, k).unwrap();

            // Run correctness check
            dev.launch_fused_dequant_gemm_iq4xs_for_ab(a_r, w_r, o_ref, m, n, k)
                .unwrap();
            dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(act_q81_r, w_r, o_dot, m, n, k)
                .unwrap();
            dev.synchronize();

            let ref_h: Vec<f32> = out_ref.to_cpu_vec_f32().unwrap();
            let dot_h: Vec<f32> = out_dot.to_cpu_vec_f32().unwrap();

            let mut max_abs = 0.0f32;
            let mut sum_sq_diff = 0.0f64;
            let mut sum_sq_ref = 0.0f64;
            for i in 0..ref_h.len() {
                let r = ref_h[i];
                let d = dot_h[i];
                let diff = (r - d).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                sum_sq_diff += (diff as f64) * (diff as f64);
                sum_sq_ref += (r as f64) * (r as f64);
            }
            let max_ref = ref_h.iter().fold(0f32, |x, v| x.max(v.abs())).max(1e-6);
            let rms_rel = (sum_sq_diff / sum_sq_ref.max(1e-12)).sqrt();
            println!(
                "Parity Shape N={} K={}: max_abs={:.6}, max_ref={:.2}, worst_rel={:.4e}, rms_rel={:.4e}",
                n,
                k,
                max_abs,
                max_ref,
                max_abs / max_ref,
                rms_rel
            );

            // Benchmark
            let iters = 50;
            // Warmup
            for _ in 0..5 {
                dev.launch_fused_dequant_gemm_iq4xs_for_ab(a_r, w_r, o_ref, m, n, k)
                    .unwrap();
                dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(act_q81_r, w_r, o_dot, m, n, k)
                    .unwrap();
            }
            dev.synchronize();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                dev.launch_fused_dequant_gemm_iq4xs_for_ab(a_r, w_r, o_ref, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let dur_grouped = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(act_q81_r, w_r, o_dot, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let dur_dot4 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let gb = total_bytes as f64 / 1e9;
            let bw_grouped = gb / (dur_grouped * 1e-6);
            let bw_dot4 = gb / (dur_dot4 * 1e-6);

            println!(
                "Speed Shape N={} K={} ({:.2} MB):\n  grouped: {:.2} us ({:.1} GB/s)\n  dot4_v2: {:.2} us ({:.1} GB/s) -> Speedup: {:.2}x",
                n,
                k,
                total_bytes as f64 / 1e6,
                dur_grouped,
                bw_grouped,
                dur_dot4,
                bw_dot4,
                dur_grouped / dur_dot4
            );
        }
    }
}
