//! Q6_K output-head parity for the Qwen3.8 Q4_K checkpoint.
//!
//! `output.weight` is Q6_K `[248320, 5120]` (1.04 GB packed) and is consumed
//! through the fused dequant-GEMM path. It is the last unverified component in
//! Qwen's first forward pass: the prompt tokens now match the Ollama reference
//! byte for byte, and the packed Q4_K embedding gather is bit-exact against the
//! host reference, yet the model still emits gibberish at token 0.
//!
//! A wrong Q6_K dequant on the head produces exactly that signature: no fault,
//! a plausible token count, meaningless text.
//!
//! The reference is the HOST `grim_quant::dequant_q6k` on the same packed bytes,
//! not another device kernel, so a shared error cannot cancel out.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{DType, Shape};
use std::sync::Arc;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

/// Row count is kept small so the test runs in milliseconds; `dim` matches the
/// real head so the same tile/quant-block geometry is exercised.
const N_ROWS: usize = 64;
const DIM: usize = 5120;

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q6k_fused_gemm_matches_host_reference() {
    let Some(dev) = gpu_device() else { return };

    // A representative weight matrix: values spanning the range Q6_K encodes,
    // with a few large magnitudes to exercise the per-block ql scale.
    let w_host: Vec<f32> = (0..N_ROWS * DIM)
        .map(|i| {
            let t = i as f32;
            (t * 0.0007).sin() * 0.8 + (t * 0.013).cos() * 0.2
        })
        .collect();
    let w_packed = grim_quant::quant_q6k(&w_host).expect("quant_q6k");
    let w_ref = grim_quant::dequant_q6k(&w_packed, N_ROWS * DIM).expect("dequant_q6k");

    // Upload the packed weights and an activation row.
    let packed_storage = dev
        .from_cpu_bytes(
            &w_packed,
            &Shape::new(vec![N_ROWS, DIM]),
            DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::Storage::Native,
            },
        )
        .expect("upload packed q6k");

    // m=1 decode-shaped GEMV: y[dim] = w @ x, exactly the output-head shape at
    // a single decode step.
    let m = 1usize;
    let a_host: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.01 - 25.0).collect();
    let a_storage = dev
        .from_cpu(&a_host, &Shape::new(vec![m, DIM]), DType::F32)
        .expect("upload activation");

    // Host reference for the same product, computed in f64.
    let mut want = vec![0.0f64; N_ROWS];
    for r in 0..N_ROWS {
        let mut acc = 0.0f64;
        for c in 0..DIM {
            acc += w_ref[r * DIM + c] as f64 * a_host[c] as f64;
        }
        want[r] = acc;
    }

    let out_shape = Shape::new(vec![m, N_ROWS]);
    let res = dev.fused_quant_gemm(
        a_storage.as_ref(),
        packed_storage.as_ref(),
        grim_tensor::QuantFormat::Q6K,
        &out_shape,
    );
    let (out, _handle) = match res {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[q6k-head] fused_quant_gemm unavailable: {e}");
            panic!("fused_quant_gemm(Q6K) must be callable: {e}");
        }
    };
    let got = out.to_cpu_vec_f32().expect("read output");
    assert_eq!(got.len(), N_ROWS);

    // Relative error against a reference that is already Q6_K-quantized, so this
    // isolates the device decode/GEMM from the quantization error itself.
    let mut worst_rel = 0.0f32;
    let mut worst_row = 0usize;
    for r in 0..N_ROWS {
        let scale = want[r].abs().max(1.0);
        let rel = ((got[r] as f64 - want[r]).abs() / scale) as f32;
        if rel > worst_rel {
            worst_rel = rel;
            worst_row = r;
        }
    }
    eprintln!(
        "[q6k-head] Q6_K fused GEMM vs host reference: worst relative error \
         {worst_rel:.6} at row {worst_row} (got {}, want {})",
        got[worst_row], want[worst_row]
    );
    assert!(
        worst_rel < 0.02,
        "Q6_K output head disagrees with the host reference: worst relative \
         error {worst_rel} at row {worst_row} (got {}, want {})",
        got[worst_row],
        want[worst_row]
    );
}
