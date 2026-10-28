use grim_backend_rocm::{RocmDevice, as_rocm};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

fn main() {
    unsafe {
        std::env::set_var("HIP_VISIBLE_DEVICES", "0,1");
        std::env::set_var("GRIM_GPU_TEST", "1");
    }

    println!("=== Q6_K F32ACT LM HEAD SPEED & PARITY PROBE ===");
    // Test on Device 1 (where LM head executes)
    let dev = RocmDevice::try_new(1).expect("RocmDevice::try_new on dev1");

    // LM head shape: N = 248320, K = 5120
    let n = 248320;
    let k = 5120;
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
    let row_bytes = n_sb * 210;
    let total_bytes = row_bytes * n;
    println!(
        "LM Head size: {:.2} MB (N={} K={})",
        total_bytes as f64 / 1e6,
        n,
        k
    );

    let mut w_bytes = vec![0u8; total_bytes];
    // Fill deterministic Q6_K weights
    for col in 0..n {
        for sb in 0..n_sb {
            let offset = col * row_bytes + sb * 210;
            // d = 0.5 in fp16 @ 208
            w_bytes[offset + 208] = 0x00;
            w_bytes[offset + 209] = 0x38;
            // scales = 1 @ 192..208
            for sc in 0..16 {
                w_bytes[offset + 192 + sc] = 1;
            }
            // ql
            for j in 0..128 {
                w_bytes[offset + j] =
                    ((col + sb + j) % 16) as u8 | ((((col + sb + j + 4) % 16) as u8) << 4);
            }
            // qh
            for j in 0..64 {
                w_bytes[offset + 128 + j] = ((col + sb + j) % 4) as u8;
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
                storage: Storage::KQuant(KQuantScheme::Q6K),
            },
        )
        .expect("upload W");

    let a_r = as_rocm(a_dev.as_ref()).unwrap();
    let w_r = as_rocm(w_dev.as_ref()).unwrap();

    let out_dev = dev
        .zeros(
            &Shape::new(vec![m, n]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )
        .unwrap();
    let o_r = as_rocm(out_dev.as_ref()).unwrap();

    // Warmup
    for _ in 0..3 {
        dev.launch_dot4_q6k_f32act_gemv_for_ab(a_r, w_r, o_r, m, n, k)
            .unwrap();
    }
    dev.synchronize();

    let iters = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        dev.launch_dot4_q6k_f32act_gemv_for_ab(a_r, w_r, o_r, m, n, k)
            .unwrap();
    }
    dev.synchronize();
    let dur_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let gb = total_bytes as f64 / 1e9;
    let bw = gb / (dur_us * 1e-6);

    println!(
        "Vectorized Q6_K f32act LM Head Execution:\n  Latency: {:.2} ms ({:.1} us)\n  Bandwidth: {:.1} GB/s",
        dur_us / 1000.0,
        dur_us,
        bw
    );
}
