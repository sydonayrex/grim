#![cfg(feature = "rocm")]

use std::time::Instant;
use grim_backend_rocm::{RocmDevice, RocmStorage};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, Shape};

fn make_q4k_bytes(n: usize, k: usize) -> Vec<u8> {
    let n_sb = k / 256;
    let total = n * n_sb * 144;
    let mut buf = vec![0u8; total];
    for col in 0..n {
        for sb in 0..n_sb {
            let off = (col * n_sb + sb) * 144;
            buf[off..off + 2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
            buf[off + 2..off + 4].copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
            for i in 0..12 {
                buf[off + 4 + i] = ((col + sb + i * 7) & 0x3F) as u8;
            }
            for i in 0..128 {
                buf[off + 16 + i] = (((col * 3 + sb * 5 + i * 11) & 0x0F)
                    | (((col * 5 + sb * 3 + i * 13) & 0x0F) << 4)) as u8;
            }
        }
    }
    buf
}

#[test]
fn wmma_q4k_iu8_parity_and_bench() {
    let dev = RocmDevice::new(0);
    let alloc = dev.allocator_handle();
    for &(m, n, k) in &[(27usize, 5120usize, 5120usize), (27, 17408, 5120), (4056, 5120, 5120)] {
        let a_host: Vec<f32> = (0..m * k)
            .map(|i| (((i * 17 + 3) % 251) as f32 - 125.0) * 0.01)
            .collect();
        let a_rocm = RocmStorage::copy_from_host(&a_host, &Shape::new(vec![m, k]), DType::F32, &alloc, 0).unwrap();

        let b_bytes = make_q4k_bytes(n, k);
        let b_dtype = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::KQuant(KQuantScheme::Q4K),
        };
        let b_rocm = RocmStorage::copy_from_host_raw_bytes(&b_bytes, &Shape::new(vec![n, k]), b_dtype, &alloc, 0).unwrap();

        let q81_bytes = m * (k / 32) * 36;
        let act_q81 = RocmStorage::alloc_gpu(&Shape::new(vec![q81_bytes]), DType::U8, &alloc, 0).unwrap();
        dev.launch_quantize_q8_1(&a_rocm, &act_q81, m, k).unwrap();

        let out_ref = RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &alloc, 0).unwrap();
        let out_wmma = RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &alloc, 0).unwrap();

        // Parity check
        dev.launch_dot4_q4k_q81_gemv_rows_r4_for_ab(&act_q81, &b_rocm, &out_ref, m, n, k).unwrap();
        dev.launch_mmq_q4k_q81_wmma_iu8_m16n64(&act_q81, &b_rocm, &out_wmma, m, n, k).unwrap();
        dev.synchronize();

        let ref_v = out_ref.to_cpu_vec_f32().unwrap();
        let wmma_v = out_wmma.to_cpu_vec_f32().unwrap();
        let mut max_diff = 0.0f32;
        for (a, b) in ref_v.iter().zip(wmma_v.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-2,
            "parity failed for M={m} N={n} K={k}: max_diff={max_diff} (ref[0]={}, wmma[0]={})",
            ref_v[0],
            wmma_v[0]
        );

        // Speed bench (10 warmup + 30 timed)
        for _ in 0..10 {
            dev.launch_dot4_q4k_q81_gemv_rows_r4_for_ab(&act_q81, &b_rocm, &out_ref, m, n, k).unwrap();
            dev.launch_mmq_q4k_q81_wmma_iu8_m16n64(&act_q81, &b_rocm, &out_wmma, m, n, k).unwrap();
        }
        dev.synchronize();

        let t0 = Instant::now();
        for _ in 0..30 {
            dev.launch_dot4_q4k_q81_gemv_rows_r4_for_ab(&act_q81, &b_rocm, &out_ref, m, n, k).unwrap();
        }
        dev.synchronize();
        let ref_ms = t0.elapsed().as_secs_f64() * 1000.0 / 30.0;

        let t1 = Instant::now();
        for _ in 0..30 {
            dev.launch_mmq_q4k_q81_wmma_iu8_m16n64(&act_q81, &b_rocm, &out_wmma, m, n, k).unwrap();
        }
        dev.synchronize();
        let wmma_ms = t1.elapsed().as_secs_f64() * 1000.0 / 30.0;

        eprintln!(
            "[wmma_iu8] M={m:4} N={n:5} K={k:5} | max_diff={max_diff:.2e} | rows_r4={ref_ms:.3} ms | wmma_iu8={wmma_ms:.3} ms | speedup={:.2}x",
            ref_ms / wmma_ms
        );
    }
}
