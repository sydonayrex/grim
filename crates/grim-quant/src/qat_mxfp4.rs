//! WI-E5 (FIND-5b): MXFP4 Quantization-Aware Training support.
//!
//! `fake_quant_mxfp4` round-trips weights through the SAME quantize/dequantize
//! code path the final real-quantize uses (`quant_mxfp4_matrix` +
//! `mxfp4_e2m1_to_f32`), guaranteeing that the trained-in quantization noise
//! matches deployment exactly. The STE (straight-through estimator) wrapper in
//! grim-autograd passes gradients through unchanged.

use crate::{mxfp4_e2m1_to_f32, quant_mxfp4_matrix};
use grim_tensor::error::{Error, Result};

fn cfg_err(msg: String) -> Error {
    // grim_tensor::Error has no Config variant; Backend carries the message.
    Error::Backend(msg)
}

/// Fake-quantize a row-major `[rows, k]` f32 matrix through MXFP4.
///
/// Each 32-element block shares one E8M0 exponent chosen identically to
/// `quant_mxfp4_matrix`, so `fake_quant_mxfp4(w)` equals dequantizing
/// `quant_mxfp4_matrix(w)` bit-for-bit — training sees exactly the values the
/// deployed kernel will compute with.
///
/// `k` must be a multiple of 32 (same constraint as the real packer).
pub fn fake_quant_mxfp4(data: &[f32], rows: usize, k: usize) -> Result<Vec<f32>> {
    if k == 0 || k % 32 != 0 || rows == 0 {
        return Err(cfg_err(format!(
            "fake_quant_mxfp4: k={k} must be a positive multiple of 32 and rows={rows} > 0"
        )));
    }
    if data.len() < rows * k {
        return Err(cfg_err(format!(
            "fake_quant_mxfp4: {} elements < {rows}x{k}",
            data.len()
        )));
    }

    let (codes, exps) = quant_mxfp4_matrix(data, rows, k);
    let exps_per_row = k / 32;
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for b in 0..exps_per_row {
            let exp_byte = exps[r * exps_per_row + b];
            for i in 0..16 {
                let byte = codes[r * (k / 2) + b * 16 + i];
                // Even element low nibble, odd high nibble — same packing as
                // quant_mxfp4_matrix.
                out.push(mxfp4_e2m1_to_f32(byte & 0x0F, exp_byte));
                out.push(mxfp4_e2m1_to_f32(byte >> 4, exp_byte));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as i32) as f32 / (i32::MAX as f32) * 0.1
            })
            .collect()
    }

    /// Round-trip exactness vs the real packer: fake_quant must equal
    /// dequant(quant(w)) on the same code path.
    #[test]
    fn fake_quant_matches_real_quant_dequant() {
        for &k in &[32usize, 256] {
            let w = deterministic(k * 3, 0xFEED);
            let faked = fake_quant_mxfp4(&w, 3, k).expect("fake_quant");
            let (codes, exps) = quant_mxfp4_matrix(&w, 3, k);
            for r in 0..3 {
                for j in 0..k {
                    let code_byte = codes[r * (k / 2) + (j / 2)];
                    let code = if j % 2 == 0 {
                        code_byte & 0x0F
                    } else {
                        code_byte >> 4
                    };
                    let exp = exps[r * (k / 32) + j / 32];
                    let real = mxfp4_e2m1_to_f32(code, exp);
                    assert_eq!(
                        faked[r * k + j],
                        real,
                        "k={k} [{r},{j}]: fake={} real={real}",
                        faked[r * k + j]
                    );
                }
            }
        }
    }

    /// STE semantics: gradient passes through unchanged. Exercised via the
    /// autograd identity-backward contract — here we assert fake-quant is a
    /// pure elementwise map of the packed domain so d(fake)/d(w) ≈ I holds
    /// away from code boundaries.
    #[test]
    fn fake_quant_is_deterministic_and_idempotent_domain() {
        let w = deterministic(256, 0xBEEF);
        let once = fake_quant_mxfp4(&w, 1, 256).expect("first");
        let twice = fake_quant_mxfp4(&once, 1, 256).expect("second");
        // Quantization is idempotent: codes are already representable.
        assert_eq!(once, twice);
    }

    #[test]
    fn rejects_bad_shapes() {
        let w = vec![0.0f32; 64];
        assert!(fake_quant_mxfp4(&w, 1, 33).is_err());
        assert!(fake_quant_mxfp4(&w[..32], 2, 32).is_err());
    }
}
