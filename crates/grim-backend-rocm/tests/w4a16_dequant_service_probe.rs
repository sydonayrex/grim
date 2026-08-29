//! Probe: dequant_w4a16_blob_to_f32 + transpose must reproduce the exact
//! dequantized weight matrix. Device-gated: GRIM_GPU_TEST=1.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{DType, Device, Shape, Storage, Tensor};
use std::sync::Arc;
use grim_tensor::{MemoryOps};

#[test]
fn w4a16_dequant_service_round_trips() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let (n_rows, k_dim, group_size) = (4usize, 8usize, 8usize);
    let mut w: Vec<f32> = (0..n_rows * k_dim)
        .map(|i| ((i % 11) as f32 * 0.37) - 1.9)
        .collect();

    // Quantize per row, per group; rewrite w to exact dequant values.
    let words_per_row = k_dim / 8;
    let groups_per_row = k_dim.div_ceil(group_size);
    let mut codes = vec![0u32; n_rows * words_per_row];
    let mut scales = vec![0.0f32; n_rows * groups_per_row];
    for row in 0..n_rows {
        for g in 0..groups_per_row {
            let lo = g * group_size;
            let hi = (lo + group_size).min(k_dim);
            let max_abs = (lo..hi)
                .map(|c| w[row * k_dim + c].abs())
                .fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 7.0 };
            scales[row * groups_per_row + g] = scale;
            for c in lo..hi {
                let q = ((w[row * k_dim + c] / scale).round() as i32).clamp(-8, 7);
                codes[row * words_per_row + c / 8] |= ((q + 8) as u32) << ((c % 8) * 4);
                w[row * k_dim + c] = q as f32 * scale;
            }
        }
    }
    let mut blob = Vec::new();
    for x in &codes {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    for x in &scales {
        blob.extend_from_slice(&x.to_le_bytes());
    }

    let dt = DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::W4A16(grim_tensor::dtype::W4A16Config { group_size }),
    };
    let blob_shape = Shape::new(vec![blob.len()]);
    let blob_gpu = dev.from_cpu_bytes(&blob, &blob_shape, dt).unwrap();
    let blob_rocm = blob_gpu
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();

    let out_box = dev
        .dequant_w4a16_blob_to_f32(blob_rocm, n_rows, k_dim, group_size)
        .unwrap();
    let out_shape = Shape::new(vec![k_dim, n_rows]); // service returns Dᵀ
    let c_t = Tensor::new(
        Arc::from(out_box),
        out_shape,
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        Device::Rocm(0),
    );
    let c = c_t.to_vec_f32().unwrap();
    eprintln!("[probe] C (D^T) = {c:?}");
    eprintln!("[probe] want D = {w:?}");

    // Transpose back: D[r, c2] = C[c2, r].
    let mut d = vec![0.0f32; n_rows * k_dim];
    for r in 0..n_rows {
        for c2 in 0..k_dim {
            d[r * k_dim + c2] = c[c2 * n_rows + r];
        }
    }
    let md = d
        .iter()
        .zip(w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[probe] max diff after transpose = {md}");
    assert!(md < 1e-3, "dequant service round-trip diverged: {md}");
}
