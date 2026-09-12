//! SPEED-DOT-OPFUSE: GPU parity for the fused RMSNorm + int8 q8_1 quantizer
//! (`grim_rmsnorm_quant_i8`) against a CPU reference of the same two steps.
//! Output layout is the packed 36-byte-per-block q8_1 format consumed by
//! `grim_dot4_q80_q81_gemv`.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test rmsnorm_quant_parity

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

#[test]
fn rmsnorm_quant_fused_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let eps = 1e-5_f32;
    for k in [1024usize, 4608usize] {
        let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
        let w: Vec<f32> = (0..k).map(|i| 1.0 + (i % 7) as f32 * 0.03).collect();

        let x_dev = CoreTensorOps::from_cpu(&dev, &x, &Shape::new(vec![k]), DType::F32)
            .expect("upload x");
        let w_dev = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![k]), DType::F32)
            .expect("upload w");
        let n_blocks = k / 32;
        let out_dev = dev
            .zeros(&Shape::new(vec![n_blocks * 36]), DType::U8)
            .expect("alloc out");

        let x_ref = x_dev
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("x is RocmStorage");
        let w_ref = w_dev
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("w is RocmStorage");
        let out_ref = out_dev
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("out is RocmStorage");

        dev.launch_rmsnorm_quant_i8(x_ref, w_ref, eps, k, out_ref, None)
            .expect("launch fused rmsnorm+quant");
        dev.synchronize();

        let got = out_dev.to_cpu_vec_f32().expect("readback");
        let got_bytes: Vec<u8> = got.iter().map(|v| *v as u8).collect();

        // The GPU reduces the sum-of-squares with a 32-lane tree (different
        // fp32 associativity than the host's sequential sum), so `d` can land
        // 1 fp16 ULP away from the host value and codes shift ±1 near
        // quantization boundaries. Correct contract: dequantized values match
        // the CPU-normalized row within one code step.
        let n_blocks = k / 32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / k as f32 + eps).sqrt();
        let mut max_err = 0.0f32;
        for blk in 0..n_blocks {
            let base = blk * 32;
            let d = half::f16::from_le_bytes([got_bytes[blk * 36], got_bytes[blk * 36 + 1]])
                .to_f32();
            for j in 0..32 {
                let code = got_bytes[blk * 36 + 4 + j] as i8;
                let deq = code as f32 * d;
                let want_v = x[base + j] * inv_rms * w[base + j];
                max_err = max_err.max((deq - want_v).abs());
            }
        }
        // One code step is d; rounding error is 0.5d; allow 1.0d for shifts.
        let tol = {
            let d0 = half::f16::from_le_bytes([got_bytes[0], got_bytes[1]]).to_f32();
            d0.max(1e-6)
        };
        eprintln!("[rmsnorm_quant] k={k} max_dequant_err={max_err:.3e} (tol {tol:.3e})");
        assert!(
            max_err <= tol,
            "fused rmsnorm+quant dequant error {max_err} exceeds one code step {tol} at k={k}"
        );
    }
}
