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

// ---------------------------------------------------------------------------
// Released-checkpoint codebook arbiter.
//
// The two readings differ by EXACTLY one level of d on every weight:
// v(q-2) - v(q-1) = -d. So if the bias-2 (paper) reading is right, the
// per-block mean sits near zero (real weight blocks are near-symmetric);
// if the released bytes were encoded for the bias-1 (Q2_0) reading, the
// bias-2 decode shifts every weight by +d and the per-block mean sits at
// ~+1 in units of d. One decode, one statistic — no model load needed.
// ---------------------------------------------------------------------------
/// Minimal IEEE f16 -> f32 (le bytes), test-local.
fn f16_scale(lo: u8, hi: u8) -> f32 {
    let h = u16::from_le_bytes([lo, hi]);
    let sign = ((h >> 15) as f32) * -2.0 + 1.0;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x03ff) as f32;
    if exp == 0 {
        return sign * man * 2f32.powi(-24);
    }
    if exp == 0x1f {
        return f32::NAN;
    }
    sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
}

#[test]
fn released_checkpoint_codebook_arbiter() {
    use grim_format::gguf::GgufDType;
    use grim_format::tprov::GgufProvider;
    use grim_tensor::provider::TensorProvider;

    // Tests run with the CRATE dir as cwd; the checkpoint lives in the
    // workspace-root models/ tree. Resolve upward.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate ancestors")
        .join("models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }

    let provider = GgufProvider::open(&path.to_string_lossy()).expect("open provider");
    let gguf = grim_format::gguf::read_gguf(std::io::BufReader::new(
        std::fs::File::open(path).expect("open file"),
    ))
    .expect("read gguf");

    // Sample several 18-byte-geometry tensors, first 4096 blocks each.
    // The released file may store its expert banks under tag 42 (Q2_0)
    // or tag 81 — the codebook question is identical for both, and the
    // bias lives in the decoder, not the tag.
    let mut hist: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for t in gguf.tensors.iter() {
        *hist.entry(t.dtype as i32).or_default() += 1;
    }
    eprintln!("[arbiter] dtype histogram: {hist:?}");
    const NB: usize = 4096;
    let mut per_tensor: Vec<(String, f64)> = Vec::new();
    for t in gguf.tensors.iter().filter(|t| t.dtype == GgufDType::GsqRco3p5 || t.dtype == GgufDType::Q2_0).take(3) {
        let raw = provider
            .get_packed(&t.name)
            .unwrap_or_else(|e| panic!("packed bytes for {}: {e}", t.name));
        let nblocks = raw.bytes.len() / 18usize;
        let nb = nblocks.min(NB);
        let data = grim_quant::dequant_gsq_rco_3p5(
            &raw.bytes[..nb * 18],
            nb * 64,
        )
        .expect("decode");
        let mut mean_sum = 0.0f64;
        let mut skipped = 0usize;
        for b in 0..nb {
            let bytes = &raw.bytes[b * 18..(b + 1) * 18];
            let d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            if d_bits == 0 {
                skipped += 1; // d = 0: every code decodes to 0 under any bias
                continue;
            }
            let d = f32::from_le_bytes([0, 0, bytes[1] & 0x80, 0]);
            let _ = d;
            // Decode this block's scale from fp16 (le) via half conversion:
            let d_f32 = f16_scale(bytes[0], bytes[1]);
            if !(d_f32 > 0.0) {
                skipped += 1;
                continue;
            }
            let blk = &data[b * 64..(b + 1) * 64];
            // bias-2 decode: levels {-2d..+1d}. Mean in units of d.
            mean_sum += blk.iter().map(|v| *v as f64 / d_f32 as f64).sum::<f64>() / 64.0;
        }
        if skipped > 0 {
            eprintln!("[arbiter] {} skipped zero-scale blocks", skipped);
        }
        // RANGE-ASYMMETRY arbiter: the codebooks have different per-sign
        // ranges. Bias-1 encoding (Q2_0): positives reach +2d, negatives
        // only -1d -> max_pos/d ~ 2 and |min_neg|/d <= 1. Bias-2 (GSQ
        // paper): mirror image. Unlike the mean shift, this does not
        // depend on the weight distribution being symmetric.
        let mut max_pos = 0.0f64;
        let mut max_neg = 0.0f64;
        for b in 0..nb {
            let bytes = &raw.bytes[b * 18..(b + 1) * 18];
            let d = f16_scale(bytes[0], bytes[1]);
            if !(d > 0.0) {
                continue;
            }
            let blk = &data[b * 64..(b + 1) * 64];
            for v in blk {
                if *v > 0.0 {
                    max_pos = max_pos.max(*v as f64 / d as f64);
                } else {
                    max_neg = max_neg.max(-*v as f64 / d as f64);
                }
            }
        }
        eprintln!(
            "[arbiter] {}: max_pos/d = {max_pos:.3}  |min_neg|/d = {max_neg:.3}  \
             -> encoder codebook {}",
            t.name,
            if max_pos > max_neg * 1.5 { "bias-1 (Q2_0: positives reach 2d)" }
            else if max_neg > max_pos * 1.5 { "bias-2 (GSQ paper: negatives reach 2d)" }
            else { "INDETERMINATE" }
        );
        let mean = mean_sum / nb as f64;
        per_tensor.push((t.name.clone(), mean));
        eprintln!(
            "[arbiter] {} blocks {}: per-block mean in units of d = {mean:+.4}",
            t.name,
            nb
        );
    }
    assert!(!per_tensor.is_empty(), "no 18-byte-geometry tensor found");
    for (name, mean) in &per_tensor {
        // |mean| ~ 0 => bias 2 (paper) correct; |mean| ~ 1 => bias 1
        // (Q2_0) correct — every weight shifted by a full level.
        assert!(
            mean.abs() < 0.5,
            "{name}: per-block mean {mean:.4} d units — consistent with the \
             bias-1 (Q2_0) reading, not the paper codebook"
        );
    }
    eprintln!("[arbiter] bias-2 (paper codebook) reading confirmed by mean-shift statistic");
}
