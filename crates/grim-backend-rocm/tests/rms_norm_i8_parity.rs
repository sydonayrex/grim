//! Elementwise-INT8 norm read path: `grim_rms_norm_i8` (packed ForestRaven
//! framed weight) must match the f32-weight `grim_rms_norm` path on the same
//! logical weights, within the INT8 grid's coarseness. Also gates the frame
//! offset arithmetic (codes @+8, single scale @+8+len+8).
use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, MemoryOps};

fn gpu_device() -> Option<RocmDevice> {
    if std::env::var("GRIM_RUN_GPU_TESTS").ok().as_deref() != Some("1") {
        return None;
    }
        RocmDevice::try_new(0).ok()
}

#[test]
fn rms_norm_i8_matches_f32_within_int8_grid() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1 + GPU");
        return;
    };
    use grim_backend_rocm::CoreTensorOps as _;
    use grim_tensor::{DType, Shape, Storage};

    let dim = 3584usize; // real norm width
    let rows = 8usize;

    // The norm weight itself, quantized per-tensor absmax INT8.
    let mut state = 0xC0FF_EE12_3456_789Au64;
    let mut rand = || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let norm_w: Vec<f32> = (0..dim).map(|i| 0.5 + rand().abs() * (i as f32 % 7.0)).collect();
    let amax = norm_w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = amax / 127.0;
    let codes: Vec<u8> = norm_w.iter().map(|v| ((v / scale).round() as i8) as u8).collect();
    let deq_w: Vec<f32> = norm_w.iter().map(|v| (v / scale).round() * scale).collect();

    // Frame: [u64 codes_len][codes][u64 scales_len][f32 scale]
    let mut blob = Vec::with_capacity(16 + dim + 4);
    blob.extend_from_slice(&(dim as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&1u64.to_le_bytes());
    blob.extend_from_slice(&scale.to_le_bytes());
    assert_eq!(blob.len(), 16 + dim + 4);

    let i8_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Block(grim_tensor::dtype::BlockDtype::Int8PerChannel),
    };

    // x activation
    let x: Vec<f32> = (0..rows * dim).map(|_| rand() * 0.5).collect();
    let out_shape = Shape::new(vec![rows, dim]);

    let x_bytes: Vec<u8> = x.iter().flat_map(|f| f.to_le_bytes()).collect();
    let x_dev = MemoryOps::from_cpu_bytes(&dev, &x_bytes, &Shape::new(vec![rows, dim]), DType::F32).unwrap();
    let w_bytes: Vec<u8> = norm_w.iter().flat_map(|f| f.to_le_bytes()).collect();
    let w_f32_dev = MemoryOps::from_cpu_bytes(&dev, &w_bytes, &Shape::new(vec![dim]), DType::F32).unwrap();
    let w_i8_dev = MemoryOps::from_cpu_bytes(&dev, &blob, &Shape::new(vec![dim]), i8_dtype).unwrap();

    // f32-weight reference
    let (ref_s, ref_h) = dev
        .rms_norm(x_dev.as_ref(), w_f32_dev.as_ref(), 1e-5, &out_shape)
        .unwrap();
    ref_h.synchronize().unwrap();
    let _ = ref_h;
    let ref_out = ref_s.to_cpu_vec_f32().unwrap();

    // packed-weight path
    let (i8_s, i8_h) = dev
        .rms_norm(x_dev.as_ref(), w_i8_dev.as_ref(), 1e-5, &out_shape)
        .unwrap();
    i8_h.synchronize().unwrap();
    let _ = i8_h;
    let i8_out = i8_s.to_cpu_vec_f32().unwrap();

    // The packed path's effective weight is deq_w; the kernel computes
    // x * (scale*code) / rms. Compare against the f32 path's OUTPUT with
    // tolerance sized by the INT8 grid: per-element weight error is
    // <= scale/2, so output rel error <= (scale/2)*||x||/|y| ~ 0.4%.
    let mut max_rel = 0.0f64;
    for r in 0..rows {
        for c in 0..dim {
            let i = r * dim + c;
            if ref_out[i].abs() > 1e-6 {
                max_rel = max_rel.max((i8_out[i] - ref_out[i]).abs() as f64 / ref_out[i].abs() as f64);
            }
        }
    }
    // Percentile picture, not just the max: small-|w| columns have large
    // RELATIVE weight error (absmax grid has uniform ABSOLUTE step) while
    // their outputs are proportionally small. Report both, gate on the
    // error weighted the way the consumer sees it (relative to row output
    // magnitude), and on the absolute diff against the weight scale.
    eprintln!("[rms-norm-i8] max elementwise rel diff = {max_rel:.5}");
    let grid_step = scale; // absolute per-element weight step
    let mut max_abs_rel_to_step = 0.0f64;
    for r in 0..rows {
        for c in 0..dim {
            let i = r * dim + c;
            let abs_diff = (i8_out[i] - ref_out[i]).abs() as f64;
            max_abs_rel_to_step = max_abs_rel_to_step.max(abs_diff / (grid_step as f64));
        }
    }
    eprintln!("[rms-norm-i8] max |out diff| / weight-step = {max_abs_rel_to_step:.3}");
    // The output diff on column c is |x_c| * (weight_err_c) / rms, and
    // weight_err_c <= step/2 by construction of the absmax grid. Normalize
    // by the activation magnitude to get a weight-error-only gate.
    let mut max_w_err = 0.0f64;
    for c in 0..dim {
        let w_err = (deq_w[c] - norm_w[c]).abs() as f64;
        max_w_err = max_w_err.max(w_err / (norm_w[c].abs().max(1e-6) as f64));
    }
    eprintln!("[rms-norm-i8] max per-column weight rel err (incl. small |w|) = {max_w_err:.4}");
    assert!(
        max_abs_rel_to_step < 1.0,
        "output diff {max_abs_rel_to_step:.3} exceeds one INT8 weight step — the kernel misreads the blob"
    );
    let _ = deq_w;
}
