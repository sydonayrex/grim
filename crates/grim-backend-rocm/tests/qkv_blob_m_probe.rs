//! Probe: does the fused Q8_0 QKV blob GEMV survive M>1 (prefill-shaped)?
use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::RocmStorage;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape, Storage};

#[test]
#[ignore]
fn blob_gemv_m12() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let dev = box_dev();
    let (n_total, k, m) = (3072usize, 1024usize, 12usize);
    let act: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let act_s = dev
        .from_cpu(&act, &Shape::new(vec![m, k]), DType::F32)
        .unwrap();
    let q81_bytes = (k / 32) * 36 * m;
    let q81 = dev
        .alloc_storage(&Shape::new(vec![q81_bytes]), DType::F32)
        .unwrap();
    let act_r = act_s.as_any().downcast_ref::<RocmStorage>().unwrap();
    let q81_r = q81.as_any().downcast_ref::<RocmStorage>().unwrap();
    dev.launch_quantize_q8_1(act_r, q81_r, m, k).unwrap();
    let w: Vec<f32> = (0..n_total * k)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let mut blob = vec![0u8; n_total * k / 32 * 34];
    for (blk, chunk) in w.chunks(32).enumerate() {
        let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = (amax / 127.0).max(1e-30);
        let bits = half::f16::from_f32(d).to_bits();
        let o = &mut blob[blk * 34..blk * 34 + 34];
        o[0] = (bits & 0xFF) as u8;
        o[1] = ((bits >> 8) & 0xFF) as u8;
        for (j, &v) in chunk.iter().enumerate() {
            o[2 + j] = ((v / d).round().clamp(-127.0, 127.0) as i8) as u8;
        }
    }
    let blob_s = dev
        .from_cpu_bytes(
            &blob,
            &Shape::new(vec![n_total]),
            DType {
                arith: grim_tensor::ArithType::F32,
                storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
            },
        )
        .unwrap();
    let out_s = dev
        .alloc_storage(&Shape::new(vec![m, n_total]), DType::F32)
        .unwrap();
    eprintln!("probe: launching blob GEMV m={m} n={n_total} k={k}");
    let act_r2 = act_s.as_any().downcast_ref::<RocmStorage>().unwrap();
    let blob_r = blob_s.as_any().downcast_ref::<RocmStorage>().unwrap();
    let out_r = out_s.as_any().downcast_ref::<RocmStorage>().unwrap();
    match dev.launch_fused_qkv_dot4_into(act_r2, blob_r, out_r, n_total, 0, k) {
        Ok(_) => {
            dev.synchronize();
        }
        Err(e) => {
            eprintln!("probe: launch err: {e}");
            return;
        }
    }
    let out = out_s.to_cpu_vec_f32().unwrap();
    eprintln!("probe: survived, out[0..4]={:?}", &out[..4.min(out.len())]);
}

fn box_dev() -> std::sync::Arc<RocmDevice> {
    std::sync::Arc::new(RocmDevice::try_new(0).unwrap())
}
