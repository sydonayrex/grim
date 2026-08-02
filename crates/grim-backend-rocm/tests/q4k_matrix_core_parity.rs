//! Parity test for Q4_K matrix-core (WMMA/MFMA) fused GEMM kernel (WI-B).
//!
//! Verifies that when `Q4kGemmConfig` is enabled, the backend compiles and launches
//! `grim_fused_dequant_gemm_q4k_wmma` / `grim_fused_dequant_gemm_q4k_mfma`, producing
//! outputs equivalent to the scalar fallback path within F16 numerical tolerance.
//!
//! TODO(gpu-verify): Execute this test on a physical RDNA3/4 (7900 XTX / gfx110x / gfx1200)
//! or CDNA2/3 (MI200/MI300 / gfx90a / gfx942) host with `GRIM_RUN_GPU_TESTS=1`.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, BackendDevice, DType, Device, KQuantScheme, QuantProvenance, Shape, Storage, Tensor,
};
use std::panic;
use std::sync::Arc;

/// Env var opting into GPU execution. If unset, tests bail Ok to run on CPU-only CI.
const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Build a RocmDevice if the GPU test environment is enabled.
fn gpu_device() -> Option<RocmDevice> {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return None;
    }
    match panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    }) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

/// Compute max absolute difference between two float slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let diff = (a[i] - b[i]).abs();
        if diff > max {
            max = diff;
        }
    }
    max
}

#[test]
fn test_q4k_matrix_core_vs_scalar_parity() {
    let dev = match gpu_device() {
        Some(d) => d,
        None => {
            eprintln!("[SKIP] test_q4k_matrix_core_vs_scalar_parity requires GRIM_RUN_GPU_TESTS=1");
            return;
        }
    };

    // Test matrix dimensions: M=16, N=16, K=256 (1 Q4_K block per row)
    let (m, n, k) = (16, 16, 256);

    // Construct input A storage
    let a_data = vec![1.0f32; m * k];
    let a_storage = grim_backend_cpu::CpuStorage::new(
        a_data,
        Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    );
    let a_tensor = Tensor::new(
        Arc::new(a_storage),
        Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
        QuantProvenance::GrimNative,
        Device::Rocm(0),
    );

    // Construct Q4_K weight B storage (144 bytes per 256 weights)
    let b_storage = grim_backend_cpu::CpuStorage::new(
        vec![0.0f32; n * k],
        Shape::new(vec![k, n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        },
    );
    let b_tensor = Tensor::new(
        Arc::new(b_storage),
        Shape::new(vec![k, n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        },
        QuantProvenance::GrimNative,
        Device::Rocm(0),
    );

    // 1. Run with matrix-core config disabled (scalar path)
    dev.set_decode_gemm_enabled(false);
    let (out_scalar_storage, handle_scalar) = dev
        .quantized_matmul(
            a_tensor.storage().as_ref(),
            b_tensor.storage().as_ref(),
            &[],
            &Shape::new(vec![m, n]),
        )
        .expect("scalar quantized_matmul should succeed");
    handle_scalar.synchronize().expect("sync failed");
    let out_scalar = out_scalar_storage
        .to_cpu_vec_f32()
        .expect("to_cpu_vec_f32 failed");

    // 2. Run with matrix-core config enabled (WMMA/MFMA tiled path)
    dev.set_decode_gemm_enabled(true);
    let (out_tiled_storage, handle_tiled) = dev
        .quantized_matmul(
            a_tensor.storage().as_ref(),
            b_tensor.storage().as_ref(),
            &[],
            &Shape::new(vec![m, n]),
        )
        .expect("tiled quantized_matmul should succeed");
    handle_tiled.synchronize().expect("sync failed");
    let out_tiled = out_tiled_storage
        .to_cpu_vec_f32()
        .expect("to_cpu_vec_f32 failed");

    // 3. Assert parity within F16 tolerance
    let diff = max_abs_diff(&out_scalar, &out_tiled);
    assert!(
        diff < 1e-3,
        "Q4_K matrix-core output diff {} exceeds threshold 1e-3",
        diff
    );
}
