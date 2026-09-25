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
fn test_act_buffer_ordering_regression_gpu() {
    let Some(dev) = gpu_device() else {
        eprintln!("Skipping: GPU test gate off");
        return;
    };

    let m = 1usize;
    let n = 128usize;
    let k = 256usize;

    // Simulate multi-step sequential execution into shared destination buffers
    let act_vals1: Vec<f32> = (0..k).map(|i| (i as f32 * 0.1).sin()).collect();
    let act_vals2: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();

    let act1 = f32_tensor(&dev, &act_vals1, &Shape::new(vec![m, k]));
    let act2 = f32_tensor(&dev, &act_vals2, &Shape::new(vec![m, k]));

    let weight_f32: Vec<f32> = (0..n * k).map(|i| ((i * 13) % 97) as f32 / 97.0 - 0.5).collect();
    let weight_packed = pack_q80(&weight_f32, n, k);
    let weight = upload_q80(&dev, &weight_packed, n, k);

    let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));

    let act_q81 = f32_tensor(&dev, &vec![0.0f32; (k / 32) * 9], &Shape::new(vec![(k / 32) * 36]));

    // Step 1: linear_decode_into with act1
    dev.linear_decode_into(act1.as_ref(), as_rocm(weight.as_ref()).unwrap(), as_rocm(out.as_ref()).unwrap(), as_rocm(act_q81.as_ref()).unwrap()).expect("step 1");
    dev.synchronize();
    let out1 = out.to_cpu_vec_f32().unwrap();

    // Step 2: linear_decode_into with act2 reusing out buffer
    dev.linear_decode_into(act2.as_ref(), as_rocm(weight.as_ref()).unwrap(), as_rocm(out.as_ref()).unwrap(), as_rocm(act_q81.as_ref()).unwrap()).expect("step 2");
    dev.synchronize();
    let out2 = out.to_cpu_vec_f32().unwrap();

    // Verify out1 and out2 are distinct and correspond to act1 and act2 without clobber
    assert_ne!(out1, out2, "Outputs across consecutive steps should differ");
}
