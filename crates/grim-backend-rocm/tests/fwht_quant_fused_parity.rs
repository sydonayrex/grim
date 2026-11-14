use grim_backend_rocm::kernels::fwht::KERNEL_SOURCE;
use grim_quant::hadamard::fwht_signed_f32;

/// Host reference for Q8_0 block quantization (34 bytes per 32 floats: fp16 d LE + 32 i8).
fn quantize_q8_0_row_ref(input: &[f32]) -> Vec<u8> {
    assert_eq!(input.len() % 32, 0);
    let mut out = Vec::with_capacity((input.len() / 32) * 34);
    for blk in input.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 1e-9 { amax / 127.0 } else { 0.0 };
        let d_f16 = half::f16::from_f32(d.min(65504.0));
        let d_bits = d_f16.to_bits();
        let d_round = d_f16.to_f32();
        out.push((d_bits & 0xFF) as u8);
        out.push((d_bits >> 8) as u8);
        for &v in blk {
            let q = if d_round > 0.0 {
                (v / d_round).round().clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            out.push(q as u8);
        }
    }
    out
}

/// Dequantize a Q8_0 row (34 bytes per 32 elements) back to f32.
fn dequantize_q8_0_row_ref(q80: &[u8]) -> Vec<f32> {
    assert_eq!(q80.len() % 34, 0);
    let mut out = Vec::with_capacity((q80.len() / 34) * 32);
    for blk in q80.chunks_exact(34) {
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for &q in &blk[2..34] {
            out.push((q as i8 as f32) * d);
        }
    }
    out
}

#[test]
fn fwht_source_contains_fused_tiled_q80_kernel() {
    assert!(
        KERNEL_SOURCE.contains("grim_fwht_tiled_q80_f32"),
        "missing grim_fwht_tiled_q80_f32 in fwht::KERNEL_SOURCE"
    );
    assert!(
        KERNEL_SOURCE.contains("blk) * 34"),
        "grim_fwht_tiled_q80_f32 must write 34-byte Q8_0 blocks"
    );
}

/// Ported from Radiance `libr4d/r4d_selftest.cpp:2351-2425`:
/// The orthogonal Sylvester Hadamard transform satisfies <H(s*w), H(s*a)> = <w, a>
/// (with 1/sqrt(N) normalization on both sides and sign vector s in {-1, +1}^N).
/// Also verifies that outlier spikes (which ruin unrotated Q8_0 block quantization)
/// are spread across all 1024 channels before Q8_0 quantization, preserving the inner product.
#[test]
fn radiance_oracle_signed_fwht_q80_preserves_inner_product_under_outliers() {
    let n = 1024usize;
    let signs: Vec<i32> = (0..n)
        .map(|i| if (i * 2654435761usize) & 4 != 0 { -1 } else { 1 })
        .collect();

    // Activation with a massive outlier spike at channel 17 (mimics transformer hidden state outlier)
    let mut a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    a[17] = 42.0;
    let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031).cos() * 0.4).collect();

    let exact_dot: f64 = a.iter().zip(w.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();

    // Rotate both w and a with signed normalized FWHT-1024
    let mut a_rot = a.clone();
    let mut w_rot = w.clone();
    fwht_signed_f32(&mut a_rot, &signs);
    fwht_signed_f32(&mut w_rot, &signs);

    // 1. Exact f32 orthogonality oracle: <H(s*a), H(s*w)> == <a, w>
    let rot_dot: f64 = a_rot
        .iter()
        .zip(w_rot.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    assert!(
        (rot_dot - exact_dot).abs() < 1e-3,
        "orthogonal FWHT inner product drift: got {rot_dot} vs exact {exact_dot}"
    );

    // 2. Quantized Q8_0 oracle: rotated Q8_0 inner product beats unrotated Q8_0 inner product on outlier row
    let a_rot_q80 = dequantize_q8_0_row_ref(&quantize_q8_0_row_ref(&a_rot));
    let a_unrot_q80 = dequantize_q8_0_row_ref(&quantize_q8_0_row_ref(&a));
    let q80_rot_dot: f64 = a_rot_q80
        .iter()
        .zip(w_rot.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let q80_unrot_dot: f64 = a_unrot_q80
        .iter()
        .zip(w.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let err_rot = (q80_rot_dot - exact_dot).abs();
    let err_unrot = (q80_unrot_dot - exact_dot).abs();
    assert!(
        err_rot < 0.05 && err_rot < err_unrot,
        "rotated Q8_0 dot error ({err_rot}) should beat unrotated outlier error ({err_unrot})"
    );
}
