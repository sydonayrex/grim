//! WI-SB6 gate: engine-loop integration for the persistent ring.
//!
//! Submits an RMSNorm (opcode 4) followed by a column-GEMM (opcode 1) into
//! `ScytheRingExec`, drains them with ONE bounded persistent worker launch,
//! and verifies the chained output against a host reference. This is the
//! decode-loop pattern (norm → projection) executing entirely through ring
//! descriptors.
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_engine::scythe2::ScytheRingExec;
use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::{BackendDevice, BackendStorage};
use grim_tensor::{DType, Shape};

fn gpu_ready() -> bool {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return false;
    }
    true
}

fn dev_ptr(storage: &dyn BackendStorage) -> u64 {
    storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .and_then(|rs| rs.device_ptr_u64())
        .expect("rocml device pointer")
}

#[test]
fn ring_norm_then_gemm_chain_matches_host_reference() {
    if !gpu_ready() {
        return;
    }
    let m = 4usize;
    let k = 64usize;
    let n = 256usize;
    let eps = 1e-5f32;

    let mut exec = ScytheRingExec::new(16, 0).expect("ring exec");
    let dev = RocmDevice::try_new(0).expect("dev");

    let x_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.3 - 1.2).collect();
    let w_data: Vec<f32> = (0..k).map(|i| ((i % 5) as f32) * 0.2 + 0.5).collect();
    let g_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.02 - 0.12).collect();

    let x = dev
        .from_cpu(&x_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("x");
    let w = dev
        .from_cpu(&w_data, &Shape::from_slice(&[k]), DType::F32)
        .expect("w");
    let g = dev
        .from_cpu(&g_data, &Shape::from_slice(&[k, n]), DType::F32)
        .expect("g");
    let tmp = dev
        .alloc_storage(&Shape::from_slice(&[m, k]), DType::F32)
        .expect("tmp");
    let out = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("out");

    exec.submit_norm(m as u32, k as u32, dev_ptr(x.as_ref()), dev_ptr(w.as_ref()), dev_ptr(tmp.as_ref()))
        .expect("submit norm");
    exec.submit_col_gemm(
        m as u32,
        n as u32,
        k as u32,
        dev_ptr(tmp.as_ref()),
        dev_ptr(g.as_ref()),
        dev_ptr(out.as_ref()),
    )
    .expect("submit gemm");

    let drained = exec.run_batch().expect("run batch");
    assert_eq!(drained, 2, "expected both descriptors drained");

    // Host reference: RMSNorm(eps=1e-5) * weight, then x @ G.
    let got = out.to_cpu_vec_f32().expect("out readback");
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        let row = &x_data[r * k..(r + 1) * k];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / k as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        let normed: Vec<f32> =
            row.iter().zip(w_data.iter()).map(|(&v, &gw)| v * inv * gw).collect();
        for (j, cell) in want[r * n..(r + 1) * n].iter_mut().enumerate() {
            let mut acc = 0f32;
            for p in 0..k {
                acc += normed[p] * g_data[p * n + j];
            }
            *cell = acc;
        }
    }
    let d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("[sb6] ring norm->gemm chain max_abs_diff={d:.3e}");
    assert!(d < 1e-3, "SB6 ring chain diverged from host reference: {d:.3e}");

    // Ring bookkeeping must be consistent after the drain.
    assert!(exec.ring.is_empty(), "ring must be empty after run_batch");
}
