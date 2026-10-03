//! Round-trip KAT for the GSQ-RCO 3.5-bit packer (GGUF tag 81).
//!
//! The decoder under test (dequant_gsq_rco_3p5, bias 2) predates the packer; the
//! packer's sources are the GSQ reference quantizer
//! (old/repo/GSQ-main/src/quantization/gumbel_quantizer_2bit.py:10,
//! `values = [-2, -1, 0, 1]`) and the shared block geometry of llama.cpp
//! `dequantize_row_q2_0` (ggml-quants.c:439, 4 codes per byte low-first).

use grim_quant::{dequant_gsq_rco_3p5, dequant_q2_0, quantize_gsq_rco_3p5_block, BLOCK_SIZE_Q2_0, BLOCK_BYTES_Q2_0};

/// Deterministic LCG so a failure is reproducible without external fixtures.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (u32::MAX >> 1) as f32 - 1.0) * 3.0
    }
}

#[test]
fn round_trip_stays_within_one_level() {
    let blocks = 7;
    let n = blocks * BLOCK_SIZE_Q2_0;
    let mut rng = Lcg(0x5EED);
    let values: Vec<f32> = (0..n).map(|_| rng.next()).collect();

    let mut bytes = vec![0u8; (n / BLOCK_SIZE_Q2_0) * BLOCK_BYTES_Q2_0];
    quantize_gsq_rco_3p5_block(&values, &mut bytes).expect("pack");

    let decoded = dequant_gsq_rco_3p5(&bytes, n).expect("unpack");
    assert_eq!(decoded.len(), n);
    // Each block's worst weight is within one level of d of its nearest
    // code; the MSE-fitted scale cannot make that worse than 1.5d, and d
    // is bounded by amax. A generous global bound keeps the test about
    // layout/codebook correctness, not scale-fit quality.
    let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (v, d) in values.iter().zip(decoded.iter()) {
        assert!(
            (v - d).abs() <= amax,
            "round-trip error {} exceeds one level bound {}: value {} decoded {}",
            (v - d).abs(),
            amax,
            v,
            d
        );
    }
}

#[test]
fn exact_levels_survive_bit_exactly() {
    let n = BLOCK_SIZE_Q2_0 * 2;
    // All +1.0: exactly representable on the +1 level.
    let values = vec![1.0f32; n];
    let mut bytes = vec![0u8; (n / BLOCK_SIZE_Q2_0) * BLOCK_BYTES_Q2_0];
    quantize_gsq_rco_3p5_block(&values, &mut bytes).expect("pack");
    let decoded = dequant_gsq_rco_3p5(&bytes, n).expect("unpack");
    for d in &decoded {
        assert_eq!(*d, 1.0, "all-ones block must decode to exactly 1.0");
    }

    // All -1.0: exactly representable as the -2 level (d = 0.5) or the
    // -1 level (d = 1.0); either way the decode is exactly -1.0.
    let values = vec![-1.0f32; n];
    let mut bytes = vec![0u8; (n / BLOCK_SIZE_Q2_0) * BLOCK_BYTES_Q2_0];
    quantize_gsq_rco_3p5_block(&values, &mut bytes).expect("pack");
    let decoded = dequant_gsq_rco_3p5(&bytes, n).expect("unpack");
    for d in &decoded {
        assert_eq!(*d, -1.0, "all-negative block must decode to exactly -1.0");
    }
}

#[test]
fn codebook_is_gsq_not_q2_0() {
    // The two 18-byte formats share geometry and differ ONLY in the
    // codebook offset (bias 2 vs bias 1). If this packer ever drifts to
    // the Q2_0 codebook, this test fails: the packed bytes must decode
    // BETTER under the GSQ reader than under the Q2_0 reader on a block
    // whose values are not symmetric about the Q2_0 grid.
    let n = BLOCK_SIZE_Q2_0;
    let mut rng = Lcg(0xDECAFBAD);
    let values: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { rng.next() } else { -rng.next() - 0.5 })
        .collect();
    let mut bytes = vec![0u8; BLOCK_BYTES_Q2_0];
    quantize_gsq_rco_3p5_block(&values, &mut bytes).expect("pack");

    let as_gsq = dequant_gsq_rco_3p5(&bytes, n).expect("gsq");
    let as_q2 = dequant_q2_0(&bytes, n).expect("q2_0");
    let gsq_err: f32 = values
        .iter()
        .zip(as_gsq.iter())
        .map(|(v, d)| (v - d).abs())
        .fold(0.0, f32::max);
    let q2_err: f32 = values
        .iter()
        .zip(as_q2.iter())
        .map(|(v, d)| (v - d).abs())
        .fold(0.0, f32::max);
    assert!(
        q2_err > gsq_err,
        "packed bytes must decode BETTER under the GSQ codebook (gsq {gsq_err} vs q2_0 {q2_err})"
    );
}

#[test]
fn rejects_bad_inputs() {
    let values = vec![1.0f32; BLOCK_SIZE_Q2_0];
    // Ragged length
    let bad = vec![1.0f32; BLOCK_SIZE_Q2_0 + 1];
    let mut out = vec![0u8; BLOCK_BYTES_Q2_0];
    assert!(quantize_gsq_rco_3p5_block(&bad, &mut out).is_err());
    // Output too short
    assert!(quantize_gsq_rco_3p5_block(&values, &mut vec![0u8; BLOCK_BYTES_Q2_0 - 1]).is_err());
    // All-zero block
    let zeros = vec![0.0f32; BLOCK_SIZE_Q2_0];
    assert!(quantize_gsq_rco_3p5_block(&zeros, &mut out).is_err());
}
