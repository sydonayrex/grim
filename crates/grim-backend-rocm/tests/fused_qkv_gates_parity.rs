//! Phase 2 G4b gate: one fused Q8_0 QKV+gate GEMV exposes zero-copy gate slices.

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{
    ArithType, BackendStorage, CoreTensorOps, DType, MemoryOps, QuantFormat, Shape, Storage,
};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn pack_q80(values: &[f32], rows: usize, hidden: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * hidden);
    assert_eq!(hidden % 32, 0);
    let block_bytes = 34;
    let row_bytes = (hidden / 32) * block_bytes;
    let mut packed = vec![0u8; rows * row_bytes];
    for row in 0..rows {
        for block in 0..(hidden / 32) {
            let base = block * 32;
            let mut max_abs = 0.0f32;
            for i in 0..32 {
                max_abs = max_abs.max(values[row * hidden + base + i].abs());
            }
            let scale = (max_abs / 127.0).max(1e-30);
            let offset = row * row_bytes + block * block_bytes;
            let scale_bits = half::f16::from_f32(scale).to_bits();
            packed[offset] = (scale_bits & 0xff) as u8;
            packed[offset + 1] = (scale_bits >> 8) as u8;
            for i in 0..32 {
                let code = (values[row * hidden + base + i] / scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8;
                packed[offset + 2 + i] = code as u8;
            }
        }
    }
    packed
}

fn upload_q80(
    dev: &RocmDevice,
    values: &[f32],
    rows: usize,
    hidden: usize,
) -> Box<dyn BackendStorage> {
    let q80 = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
    };
    MemoryOps::from_cpu_bytes(
        dev,
        &pack_q80(values, rows, hidden),
        &Shape::new(vec![rows, hidden]),
        q80,
    )
    .expect("upload Q8_0 weight")
}

fn max_diff(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len());
    got.iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn quantize_q80(
    dev: &RocmDevice,
    weight: &dyn BackendStorage,
) -> grim_tensor::error::Result<Box<dyn BackendStorage>> {
    let (quantized, handle) = dev.quantize_on_device(weight, QuantFormat::Q8_0)?;
    handle.synchronize()?;
    Ok(quantized)
}

#[test]
#[ignore = "GPU-only G4b Phase 2 parity; run with GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test fused_qkv_gates_parity -- --ignored"]
fn fused_qkv_gates_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();

    const HIDDEN: usize = 128;
    const N_Q: usize = 64;
    const N_KV: usize = 32;
    const N_GATE: usize = 16;

    let mut seed = 0x42u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32 / u32::MAX as f32 - 0.5) * 0.5
    };
    let act_values: Vec<f32> = (0..HIDDEN).map(|_| rand()).collect();
    let wq_values: Vec<f32> = (0..N_Q * HIDDEN).map(|_| rand()).collect();
    let wk_values: Vec<f32> = (0..N_KV * HIDDEN).map(|_| rand()).collect();
    let wv_values: Vec<f32> = (0..N_KV * HIDDEN).map(|_| rand()).collect();
    let gb_values: Vec<f32> = (0..N_GATE * HIDDEN).map(|_| rand()).collect();
    let gw_values: Vec<f32> = (0..N_GATE * HIDDEN).map(|_| rand()).collect();
    let gf_values: Vec<f32> = (0..N_GATE * HIDDEN).map(|_| rand()).collect();

    let act = CoreTensorOps::from_cpu(&dev, &act_values, &Shape::new(vec![1, HIDDEN]), DType::F32)
        .expect("upload activation");
    let q81_storage = dev
        .zeros(
            &Shape::new(vec![(HIDDEN / 32) * 36]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("allocate Q8_1 activation");
    let q81 = as_rocm(q81_storage.as_ref()).expect("Q8_1 storage");
    let act_rocm = as_rocm(act.as_ref()).expect("activation storage");
    dev.launch_quantize_q8_1(act_rocm, q81, 1, HIDDEN)
        .expect("quantize activation");

    let wq = upload_q80(&dev, &wq_values, N_Q, HIDDEN);
    let wk = upload_q80(&dev, &wk_values, N_KV, HIDDEN);
    let wv = upload_q80(&dev, &wv_values, N_KV, HIDDEN);
    let w_gb = CoreTensorOps::from_cpu(
        &dev,
        &gb_values,
        &Shape::new(vec![N_GATE, HIDDEN]),
        DType::F32,
    )
    .expect("upload gb weight");
    let w_gw = CoreTensorOps::from_cpu(
        &dev,
        &gw_values,
        &Shape::new(vec![N_GATE, HIDDEN]),
        DType::F32,
    )
    .expect("upload gw weight");
    let w_gf = CoreTensorOps::from_cpu(
        &dev,
        &gf_values,
        &Shape::new(vec![N_GATE, HIDDEN]),
        DType::F32,
    )
    .expect("upload gf weight");

    let q_gb = quantize_q80(&dev, w_gb.as_ref()).expect("quantize gb reference");
    let q_gw = quantize_q80(&dev, w_gw.as_ref()).expect("quantize gw reference");
    let q_gf = quantize_q80(&dev, w_gf.as_ref()).expect("quantize gf reference");
    let q_gb = as_rocm(q_gb.as_ref()).expect("gb reference storage");
    let q_gw = as_rocm(q_gw.as_ref()).expect("gw reference storage");
    let q_gf = as_rocm(q_gf.as_ref()).expect("gf reference storage");

    let gb_ref_storage = dev
        .zeros(&Shape::new(vec![N_GATE]), DType::F32)
        .expect("allocate gb reference output");
    let gw_ref_storage = dev
        .zeros(&Shape::new(vec![N_GATE]), DType::F32)
        .expect("allocate gw reference output");
    let gf_ref_storage = dev
        .zeros(&Shape::new(vec![N_GATE]), DType::F32)
        .expect("allocate gf reference output");
    let gb_ref = as_rocm(gb_ref_storage.as_ref()).expect("gb reference ROCm storage");
    let gw_ref = as_rocm(gw_ref_storage.as_ref()).expect("gw reference ROCm storage");
    let gf_ref = as_rocm(gf_ref_storage.as_ref()).expect("gf reference ROCm storage");

    dev.linear_decode_into(act.as_ref(), q_gb, gb_ref, q81)
        .expect("independent gb GEMV")
        .synchronize()
        .expect("synchronize gb GEMV");
    dev.linear_decode_into(act.as_ref(), q_gw, gw_ref, q81)
        .expect("independent gw GEMV")
        .synchronize()
        .expect("synchronize gw GEMV");
    dev.linear_decode_into(act.as_ref(), q_gf, gf_ref, q81)
        .expect("independent gf GEMV")
        .synchronize()
        .expect("synchronize gf GEMV");

    let fused = dev
        .build_fused_gate_qkv_q80(
            wq.as_ref(),
            wk.as_ref(),
            wv.as_ref(),
            w_gb.as_ref(),
            w_gw.as_ref(),
            w_gf.as_ref(),
        )
        .expect("build fused QKV+gate blob");
    let logits = dev
        .fused_qkv_gates_dot4(q81, &fused)
        .expect("run fused QKV+gate GEMV");
    dev.synchronize();

    let output_ptr = logits
        .output
        .device_ptr()
        .expect("fused output device pointer");
    assert_eq!(
        logits.gb.device_ptr_u64(),
        output_ptr + fused.gb_offset() as u64 * 4,
        "gb view must alias fused output"
    );
    assert_eq!(
        logits.gw.device_ptr_u64(),
        output_ptr + fused.gw_offset() as u64 * 4,
        "gw view must alias fused output"
    );
    assert_eq!(
        logits.gf.device_ptr_u64(),
        output_ptr + fused.gf_offset() as u64 * 4,
        "gf view must alias fused output"
    );

    let gb = logits.gb.to_cpu_vec_f32().expect("download gb slice");
    let gw = logits.gw.to_cpu_vec_f32().expect("download gw slice");
    let gf = logits.gf.to_cpu_vec_f32().expect("download gf slice");
    let gb_ref = gb_ref_storage
        .to_cpu_vec_f32()
        .expect("download gb reference");
    let gw_ref = gw_ref_storage
        .to_cpu_vec_f32()
        .expect("download gw reference");
    let gf_ref = gf_ref_storage
        .to_cpu_vec_f32()
        .expect("download gf reference");

    let gb_diff = max_diff(&gb, &gb_ref);
    let gw_diff = max_diff(&gw, &gw_ref);
    let gf_diff = max_diff(&gf, &gf_ref);
    eprintln!(
        "[fused-qkv-gates-parity] hidden={HIDDEN} gb={gb_diff:.6} gw={gw_diff:.6} gf={gf_diff:.6}"
    );
    assert!(gb_diff <= 1e-2, "gb slice diverges: {gb_diff}");
    assert!(gw_diff <= 1e-2, "gw slice diverges: {gw_diff}");
    assert!(gf_diff <= 1e-2, "gf slice diverges: {gf_diff}");
}
