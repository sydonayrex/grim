//! DEBUG probe for the dot4 Q4_K q81 GEMV -inf (pre-existing red).
use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage,
};

#[test]
#[ignore]
fn dot4_q4k_debug_dump() {
    let dev = RocmDevice::try_new(0).unwrap();
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }
    let m = 1usize;
    let n = 128usize;
    let k = 256usize;
    let mut seed = 0x1337u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let a_f32: Vec<f32> = (0..m * k).map(|_| rand()).collect();
    // B is [N][K] row-major (the kernel reads column `col` at col*row_bytes),
    // so pack per column exactly like tests/dot4_q4k_arch_probe.rs does.
    let row_bytes = (k / 256) * 144;
    let mut b_bytes = vec![0u8; n * row_bytes];
    for col in 0..n {
        let b_col: Vec<f32> = (0..k).map(|_| rand()).collect();
        let packed = grim_quant::quant_q4k(&b_col).unwrap();
        b_bytes[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let q4k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let b_dev =
        MemoryOps::from_cpu_bytes(&dev, &b_bytes, &Shape::new(vec![b_bytes.len()]), q4k_dtype)
            .unwrap();
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let (c_storage, handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &Shape::new(vec![m, n]),
        )
        .unwrap();
    handle.synchronize().unwrap();
    let got = c_storage.to_cpu_vec_f32().unwrap();

    let row_bytes = (k / 256) * 144;
    for col in 0..3 {
        let brow = &b_bytes[col * row_bytes..(col + 1) * row_bytes];
        let b_deq = grim_quant::dequant_q4k(brow, k).unwrap();
        let acc: f32 = (0..k).map(|kk| a_f32[kk] * b_deq[kk]).sum();
        eprintln!(
            "[q4k-dbg] quantized_matmul col {col}: got={:?} want={acc:.4}",
            &got[col..(col + 4).min(got.len())]
        );
    }

    // DIRECT launch (arch-probe path): quantize A ourselves, launch the kernel
    // directly. Isolates quantized_matmul's plumbing (act buffer, fence) from
    // the kernel.
    let act = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * (k / 32) * 36]), DType::F32).unwrap();
    dev.launch_quantize_q8_1(
        grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap(),
        grim_backend_rocm::as_rocm(act.as_ref()).unwrap(),
        m,
        k,
    )
    .unwrap();
    dev.synchronize();
    let out2 = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).unwrap();
    dev.launch_dot4_q4k_q81_gemv_for_ab(
        grim_backend_rocm::as_rocm(act.as_ref()).unwrap(),
        grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap(),
        grim_backend_rocm::as_rocm(out2.as_ref()).unwrap(),
        m,
        n,
        k,
    )
    .unwrap();
    let got2 = out2.to_cpu_vec_f32().unwrap();
    for col in 0..3 {
        let brow = &b_bytes[col * row_bytes..(col + 1) * row_bytes];
        let b_deq = grim_quant::dequant_q4k(brow, k).unwrap();
        let acc: f32 = (0..k).map(|kk| a_f32[kk] * b_deq[kk]).sum();
        eprintln!(
            "[q4k-dbg] direct       col {col}: got={:?} want={acc:.4}",
            &got2[col..(col + 4).min(got2.len())]
        );
    }
}
