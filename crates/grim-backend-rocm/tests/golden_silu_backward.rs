//! Golden parity test for the fused SwiGLU backward kernel vs CPU reference.
//!
//! Mirrors the structure of `golden_q4k_gpu_mutation.rs`: builds a tiny
//! ROCm device, runs the GPU kernel on random input, and asserts elementwise
//! agreement with a CPU reference implementation.

use grim_tensor::{BackendDevice, Shape};

fn silu_backward_ref(e: &[f32], g: &[f32], dw: &[f32]) -> (Vec<f32>, Vec<f32>) {
    // Forward:  y = silu(e) * g
    //   where silu(e) = e * sigmoid(e), sigmoid(e) = 1/(1+exp(-e))
    // Backward (dw = dL/dy):
    //   df = dL/dg = silu(e) * dw          — gradient w.r.t. g (up)
    //   de = dL/de = g * silu'(e) * dw     — gradient w.r.t. e (gate)
    //         where silu'(e) = sigmoid(e) * (1 + e*(1 - sigmoid(e)))
    let n = e.len();
    let mut df = vec![0.0f32; n];
    let mut de = vec![0.0f32; n];
    for i in 0..n {
        let se = 1.0 / (1.0 + (-e[i]).exp());
        let silu_e = se * e[i];
        let dsilu = se * (1.0 + e[i] * (1.0 - se));
        df[i] = dw[i] * silu_e;
        de[i] = dw[i] * g[i] * dsilu;
    }
    (df, de)
}

#[test]
fn silu_backward_analytic_zero_input() {
    // e=0 → silu(0) = 0, sigmoid(0) = 0.5, silu'(0) = 0.5
    // df = dL/dg = silu(0) * dw      = 0 * 1 = 0
    // de = dL/de = g * silu'(0) * dw = 1 * 0.5 * 1 = 0.5
    let (df, de) = silu_backward_ref(&[0.0], &[1.0], &[1.0]);
    assert!((df[0] - 0.0).abs() < 1e-6);
    assert!((de[0] - 0.5).abs() < 1e-6);
}

#[test]
fn silu_backward_gpu_matches_reference() {
    let device = match grim_backend_rocm::RocmDevice::try_new(0).ok() {
        Some(d) => d,
        None => {
            eprintln!("no ROCm device; skipping GPU parity");
            return;
        }
    };
    let n = 4096usize;
    let e: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 2.0).collect();
    let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).cos()).collect();
    let dw: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.001).fract()).collect();

    let shape = Shape::new(vec![n]);
    let storage_e = device
        .from_cpu(&e, &shape, grim_tensor::DType::F32)
        .unwrap();
    let storage_g = device
        .from_cpu(&g, &shape, grim_tensor::DType::F32)
        .unwrap();
    let storage_dw = device
        .from_cpu(&dw, &shape, grim_tensor::DType::F32)
        .unwrap();

    let (df_storage, de_storage, _handle) =
        match device.silu_mul_backward(&*storage_e, &*storage_g, &*storage_dw, &shape) {
            Ok(v) => v,
            Err(grim_tensor::error::Error::Unimplemented(_)) => {
                eprintln!("silu_mul_backward not implemented on this backend; skipping");
                return;
            }
            Err(e) => panic!("silu_mul_backward failed: {e}"),
        };

    let df_gpu = df_storage.to_cpu_vec_f32().unwrap();
    let de_gpu = de_storage.to_cpu_vec_f32().unwrap();

    let (df_ref, de_ref) = silu_backward_ref(&e, &g, &dw);

    let max_df = df_gpu
        .iter()
        .zip(df_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_de = de_gpu
        .iter()
        .zip(de_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_df < 1e-4, "df GPU-vs-ref max abs error {max_df}");
    assert!(max_de < 1e-4, "de GPU-vs-ref max abs error {max_de}");
}
