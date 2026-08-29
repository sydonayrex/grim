//! Device verifier for the IQ4XS backward kernel (P4 red-to-green anchor).
//!
//! Mirrors `quant_backward_gpu.rs`: build dY + packed B on device,
//! call `dev.quantized_matmul_backward_dx` with `KQuantScheme::IQ4XS`,
//! and compare the device dX against a host reference via RMS relative error.
//!
//! Contract being verified (implement.md §P4 "test hook"):
//!   dX[m][k] = Σₙ dY[m][n] · B[n][k]
//! where B is IQ4_XS-packed with one 136-byte superblock per 256 weights of
//! each of the N rows (`row_bytes = (K/256)·136`, so K must be a multiple of
//! 256 for the row layout to exist — the earlier k=4 shape had
//! `blocks_per_row = 0` and verified nothing).
//!
//! The host reference decodes with the DEVICE-canonical IQ4_XS formula — the
//! same decode as `dequant_iq4xs` in `kernels/iq_gemm.rs` and
//! `dequant_iq4xs_device` in `kernels/iq_dequant.rs` (`d · sc_val · q_code`,
//! 6-bit packed sub-block scales, low/high nibble). grim-quant's
//! `dequant_iq4xs` CPU decode uses a different (codebook) formula; harmonizing
//! the two is outside P4's scope (P4 is the backward kernel's per-MAC
//! div/mod restructuring), so the reference here is an independent Rust
//! re-implementation of the device decode to pin the backward GEMM structure.
//!
//! This test is the red-to-green anchor: it fails on the per-MAC /
//! wrong-row-indexed backward kernel and passes once the P4 rewrite lands.
//!
//! Run with:
//!   GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --test p4_iq4xs_backward_device_verifier -- --ignored --nocapture

use grim_quant::quant_iq4xs;
use grim_backend_rocm::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    KQuantScheme, QuantizedMatmulBackwardResiduals, Shape,
    dtype::{ArithType, DType, Storage},
};

const MAX_RMS_REL_ERROR: f32 = 0.05;

/// f16 (IEEE 754 half) → f32 bit-exact decode. Matches
/// `fp16_to_float_device` in `shared_device_fns.rs` for normal/subnormal/
/// zero inputs (the packed `d` scales produced by `quant_iq4xs` are normal).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        let value = (mant as f32) * 2f32.powi(-24);
        if sign != 0 { -value } else { value }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Rust mirror of the DEVICE-canonical IQ4_XS single-element decode
/// (`kernels/iq_gemm.rs::dequant_iq4xs`): 136-byte superblock =
/// f16 `d` (2B) + 6-bit packed sub-block scales (6B) + 4-bit codes (128B).
fn device_decode_iq4xs(block: &[u8], in_sb: usize) -> f32 {
    debug_assert!(block.len() >= 136);
    let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let sc = &block[2..8];
    let qs = &block[8..136];
    let group = in_sb / 32;
    let sc_byte_idx = (group * 6) / 8;
    let sc_bit_offset = (group * 6) % 8;
    let mut sc_val = (sc[sc_byte_idx] >> sc_bit_offset) as u32;
    if sc_bit_offset > 2 {
        sc_val |= (sc[sc_byte_idx + 1] as u32) << (8 - sc_bit_offset);
    }
    sc_val &= 0x3F;
    let q_byte = in_sb / 2;
    let q_code = if in_sb % 2 == 0 {
        qs[q_byte] & 0x0F
    } else {
        (qs[q_byte] >> 4) & 0x0F
    };
    d * sc_val as f32 * q_code as f32
}

/// Host reference for the backward contract: decode B (packed row-major
/// [N][K], one superblock per 256 weights of each row) and compute
/// dX[m][k] = Σₙ dY[m][n] · B_deq[n][k].
fn iq4xs_backward_dx_host(dy: &[f32], packed_b: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(
        k % 256,
        0,
        "device row layout requires K to be a multiple of 256"
    );
    let row_bytes = (k / 256) * 136;
    assert_eq!(packed_b.len(), n * row_bytes);
    let mut dx = vec![0.0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut acc = 0.0f32;
            for ni in 0..n {
                let blk = &packed_b[ni * row_bytes..(ni + 1) * row_bytes];
                acc += dy[mi * n + ni] * device_decode_iq4xs(blk, ki % 256);
            }
            dx[mi * k + ki] = acc;
        }
    }
    dx
}

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
    let k = 256; // must be a multiple of 256: blocks_per_row = K/256 = 1

    let dy_host: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.25).cos()).collect();

    // B in device layout order: flat [N][K] row-major (row n holds that
    // output row's K weights, packed as K/256 superblocks per row).
    let b_flat: Vec<f32> = (0..n * k)
        .map(|i| {
            let row = i / k;
            let col = i % k;
            ((row as f32 + col as f32) * 0.05 + 0.3).cos() * 2.0
        })
        .collect();

    let dy_shape = Shape::from_slice(&[m, n]);
    let dy_rocm = dev.from_cpu(&dy_host, &dy_shape, DType::F32).unwrap();

    let packed_b = quant_iq4xs(&b_flat).unwrap();
    let b_rocm_shape = Shape::from_slice(&[b_flat.len()]);
    let b_rocm = dev
        .from_cpu_bytes(
            &packed_b,
            &b_rocm_shape,
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::IQ4XS),
            },
        )
        .unwrap();

    // Host reference decodes the SAME packed bytes with the device-canonical
    // formula (quant round-trip lossiness is identical on both sides, so the
    // comparison isolates the backward GEMM structure).
    let dx_ref = iq4xs_backward_dx_host(&dy_host, &packed_b, m, n, k);
    assert!(
        dx_ref.iter().any(|&x| x.abs() > 1e-6),
        "P4 verifier host reference must be non-trivial"
    );

    let dx_shape = Shape::from_slice(&[m, k]);
    let residuals = QuantizedMatmulBackwardResiduals::default();
    let (dx_rocm, _handle) = dev
        .quantized_matmul_backward_dx(
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

    assert_eq!(dx_device.len(), dx_ref.len());
    let rms = rms_rel_err(&dx_ref, &dx_device);
    assert!(
        rms <= MAX_RMS_REL_ERROR,
        "P4 IQ4XS backward device verifier: RMS rel err {rms:.6} exceeds limit {MAX_RMS_REL_ERROR}"
    );
}
