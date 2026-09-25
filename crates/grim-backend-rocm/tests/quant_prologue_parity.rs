use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{ArithType, CoreTensorOps, DType, MemoryOps, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn f32_tensor(
    dev: &RocmDevice,
    data: &[f32],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, shape, DType::F32).unwrap()
}

fn pack_q80(w: &[f32], rows: usize, k: usize) -> Vec<u8> {
    let blocks = k / 32;
    let mut out = vec![0u8; rows * blocks * 34];
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * 34;
            let mut amax = 0.0f32;
            for e in 0..32 {
                amax = amax.max(w[r * k + b * 32 + e].abs());
            }
            let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            let h = half::f16::from_f32(d);
            out[off..off + 2].copy_from_slice(&h.to_le_bytes());
            for e in 0..32 {
                let v = w[r * k + b * 32 + e] / d;
                out[off + 2 + e] = v.round().clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }
    out
}

fn upload_q80(
    dev: &RocmDevice,
    packed: &[u8],
    rows: usize,
    k: usize,
) -> Box<dyn grim_tensor::BackendStorage> {
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
    };
    MemoryOps::from_cpu_bytes(dev, packed, &Shape::new(vec![rows, k]), q_dtype).unwrap()
}

#[test]
#[ignore]
fn test_quant_prologue_parity_gpu() {
    let Some(dev) = gpu_device() else {
        eprintln!("Skipping: GPU test gate off");
        return;
    };

    let m = 1usize;
    let n = 128usize;
    let k = 256usize;

    let act_vals: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.1).sin()).collect();
    let act_tensor = f32_tensor(&dev, &act_vals, &Shape::new(vec![m, k]));
    let act_rocm = as_rocm(act_tensor.as_ref()).unwrap();

    let weight_f32: Vec<f32> = (0..n * k).map(|i| ((i * 7) % 113) as f32 / 113.0 - 0.5).collect();
    let weight_packed = pack_q80(&weight_f32, n, k);
    let weight_tensor = upload_q80(&dev, &weight_packed, n, k);
    let weight_rocm = as_rocm(weight_tensor.as_ref()).unwrap();

    let out_ref_tensor = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    let out_ref_rocm = as_rocm(out_ref_tensor.as_ref()).unwrap();

    let out_fused_tensor = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    let out_fused_rocm = as_rocm(out_fused_tensor.as_ref()).unwrap();

    let act_q81_tensor = f32_tensor(&dev, &vec![0.0f32; (k / 32) * 9], &Shape::new(vec![(k / 32) * 36]));
    let act_q81_rocm = as_rocm(act_q81_tensor.as_ref()).unwrap();

    // 1. Reference path: launch_quantize_q8_1 + launch_dot4_q80_q81_gemv
    dev.launch_quantize_q8_1(act_rocm, act_q81_rocm, m, k).expect("ref quantize");
    dev.launch_dot4_q80_q81_gemv(act_q81_rocm, weight_rocm, out_ref_rocm, m, n, k).expect("ref gemv");

    // 2. Fused path: launch_dot4_q80_f32act_gemv
    dev.launch_dot4_q80_f32act_gemv(act_rocm, weight_rocm, out_fused_rocm, m, n, k).expect("fused gemv");

    dev.synchronize();

    let ref_out = out_ref_tensor.to_cpu_vec_f32().unwrap();
    let fused_out = out_fused_tensor.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n {
        let diff = (ref_out[i] - fused_out[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < 1e-4,
        "Parity check failed: max_diff = {max_diff}"
    );
}
