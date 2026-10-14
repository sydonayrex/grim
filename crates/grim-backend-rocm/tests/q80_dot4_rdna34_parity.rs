//! RDNA3/4 Q8_0 dot4 GEMV parity, both the f32-act and the prequant-q81 arm,
//! at production decode shapes (k=1024; n=1024 and n=4608). The prequant arm
//! is driven through `launch_dot4_q80_q81_gemv` directly because
//! `quantized_matmul` derives K from the activation shape, which for a packed
//! q81 buffer is (k/32)*36, not k — routing a q81 buffer through it reads B out
//! of bounds. Production callers pass K explicitly.
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::QuantOps;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{CoreTensorOps, KQuantScheme, MemoryOps, Shape};

#[test]
#[ignore]
fn q80_dot4_q81_parity_on_rdna34() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    for (n, k, m) in [(1024usize, 1024usize, 1usize), (4608, 1024, 11)] {
        let mut seed = 0xA11CEu64 ^ (m as u64) ^ (n as u64);
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a_f32: Vec<f32> = (0..m * k).map(|_| rand()).collect();
        let b_f32: Vec<f32> = (0..n * k).map(|_| rand()).collect();
        let b_bytes = grim_quant::quant_q80(&b_f32).expect("quant_q80");
        let b_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &b_bytes,
            &Shape::from_slice(&[n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q80),
            },
        )
        .expect("upload q80");
        let out_shape = Shape::from_slice(&[m, n]);

        eprintln!("[q80-dot4] start m={m} n={n} k={k}");
        // ── arm 1: f32 activation (f32act kernel) ──
        let a_dev =
            CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::from_slice(&[m, k]), DType::F32).unwrap();
        unsafe { std::env::set_var("GRIM_DOT_GEMV", "1") };
        let (c1, h1) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                b_dev.as_ref(),
                &[],
                grim_tensor::QuantFormat::Q8_0,
                &out_shape,
            )
            .expect("f32act gemm");
        h1.synchronize().unwrap();
        let c1 = c1.to_cpu_vec_f32().unwrap();

        eprintln!("[q80-dot4] arm1 ok m={m}");
        // ── arm 2: q81 produced by the GPU quantizer (production path) ──
        let q81_bytes = (k / 32) * 36 * m;
        let q81_dev = dev
            .zeros(
                &Shape::new(vec![q81_bytes]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .expect("alloc q81");
        let a_rocm = a_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap();
        let q81_rocm = q81_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap();
        dev.launch_quantize_q8_1(a_rocm, q81_rocm, m, k)
            .expect("quantize q8_1");
        let out_dev = dev.zeros(&out_shape, DType::F32).expect("alloc out");
        let out_rocm = out_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap();
        let b_rocm = b_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap();
        dev.launch_dot4_q80_q81_gemv(q81_rocm, b_rocm, out_rocm, m, n, k)
            .expect("q81 gemv");
        let c2 = out_dev.to_cpu_vec_f32().unwrap();

        let row_bytes = (k / 32) * 34;
        let mut cpu = vec![0.0f32; m * n];
        for row in 0..m {
            let a_row = &a_f32[row * k..(row + 1) * k];
            for col in 0..n {
                let brow = &b_bytes[col * row_bytes..(col + 1) * row_bytes];
                let b_deq = grim_quant::dequant_q80(brow, k).unwrap();
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_row[kk] * b_deq[kk];
                }
                cpu[row * n + col] = acc;
            }
        }
        let worst = |v: &[f32]| v.iter().zip(&cpu).fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
        eprintln!(
            "[q80-dot4] m={m} n={n} k={k} f32act={:.6} q81={:.6}",
            worst(&c1),
            worst(&c2)
        );
        assert!(worst(&c1) < 0.5, "f32act arm diverges m={m} n={n}: {}", worst(&c1));
        assert!(worst(&c2) < 0.5, "q81 arm diverges m={m} n={n}: {}", worst(&c2));
    }
}
