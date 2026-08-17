//! Device verifier for the IQ4XS backward kernel (P4 red-to-green anchor).
//!
//! Mirrors `quant_backward_gpu.rs`: build dY + packed B on device,
//! call `dev.quantized_matmul_backward_dx` with `KQuantScheme::IQ4XS`,
//! and compare the device dX against a host reference (`iq4xs_backward_dx_host`
//! from `p4_iq_backward_contract.rs`) via RMS relative error.
//!
//! This runs RED on the current `iq_gemm.rs` IQ4XS backward kernel (per-MAC
//! div/mod, one dequant per MAC) and GREEN after the P4 rewrite (superblock-
//! per-thread, scale-loaded-once, no per-MAC div/mod). It is the red-to-green
//! anchor for the P4 IQ4XS backward rewrite.
//!
//! Run with:
//!   GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --test p4_iq4xs_backward_device_verifier -- --nocapture

use grim_backend_rocm::RocmDevice;
use grim_quant::quant_iq4xs;
use grim_tensor::{
    BackendDevice, KQuantScheme, QuantizedMatmulBackwardResiduals, Shape,
    dtype::{ArithType, DType, Storage},
};

const MAX_RMS_REL_ERROR: f32 = 0.05;

fn rms_rel_err(orig: &[f32], recon: &[f32]) -> f32 {
    let sum_sq: f32 = orig
        .iter()
        .zip(recon.iter())
        .map(|(o, r)| {
            let denom = o.abs().max(1e-3);
            ((o - r) / denom).powi(2)
        })
        .sum();
    (sum_sq / orig.len() as f32).sqrt()
}

fn iq4xs_backward_dx_host(dY: &[f32], b_deq: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut acc = 0.0f32;
            for ni in 0..n {
                acc += dY[mi * n + ni] * b_deq[ki * n + ni];
            }
            dx[mi * k + ki] = acc;
        }
    }
    dx
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn p4_iq4xs_backward_device_verifier_matches_host_reference() {
    let rocm_devices = match grim_backend_rocm::RocmDevice::probe() {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let dev = grim_backend_rocm::RocmDevice::try_new(rocm_devices[0].ordinal())
        .expect("RocmDevice::try_new should succeed for probed device");

    let m = 2;
    let n = 4;
    let k = 4;
    let num_weights = k * n;

    let dY_host: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32) * 0.25).cos())
        .collect();
    let b_host: Vec<f32> = (0..num_weights)
        .map(|i| {
            let row = i / n;
            let col = i % n;
            ((row as f32 + col as f32) * 0.5 + 0.3).cos() * 2.0
        })
        .collect();

    let dy_shape = Shape::from_slice(&[m, n]);
    let dy_rocm = dev.from_cpu(&dY_host, &dy_shape, DType::F32).unwrap();

    let packed_b = quant_iq4xs(&b_host).unwrap();
    let b_rocm_shape = Shape::from_slice(&[num_weights]);
    let b_rocm = dev.from_cpu_bytes(
        &packed_b,
        &b_rocm_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::IQ4XS),
        },
    )
    .unwrap();

    let host_b_deq = grim_quant::dequant_iq4xs(&packed_b, num_weights).unwrap();
    let dx_ref = iq4xs_backward_dx_host(&dY_host, &host_b_deq, m, n, k);

    let dx_shape = Shape::from_slice(&[m, k]);
    let residuals = QuantizedMatmulBackwardResiduals::default();
    let (dx_rocm, _handle) = dev.quantized_matmul_backward_dx(
        dy_rocm.as_ref(),
        b_rocm.as_ref(),
        &[],
        8, // bpw for IQ4XS
        m,
        n,
        k,
        &dx_shape,
        Some(&residuals),
    )
    .unwrap();
    let dx_device = dx_rocm.to_cpu_vec_f32().unwrap();

    let rms = rms_rel_err(&dx_ref, &dx_device);
    assert!(
        rms <= MAX_RMS_REL_ERROR,
        "P4 IQ4XS backward device verifier: RMS rel err {rms:.6} exceeds limit {MAX_RMS_REL_ERROR}"
    );
}
