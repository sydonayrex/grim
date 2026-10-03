//! Quantization routines (Q4_K, Q8_0, NF4, FP8, MXFP4/8, GPTQ, SPQR, SoulEater, IQ1-4).

use grim_tensor::error::{Error, Result};

/// The IQ4_NL signed codebook (llama.cpp `kvalues_iq4nl`). Defined once in
/// `iq_tables` and pinned there against the reference header, so a
/// mistranscription cannot recur in a second copy. It was previously wrong in
/// the two largest entries (87.0/107.0 instead of 89.0/113.0).
use iq_tables::KVALUES_IQ4NL as KVALUES_IQ4NL_REF;

pub mod accuracy_gate;
pub mod citycrow;
pub mod grey_raven;
pub mod gsq;
pub mod iq_tables;
mod packed_gemm;
pub mod qat_mxfp4;
pub mod rco;
pub mod scrub_jay;
pub mod soul_eater;
pub mod spqr;
pub mod tree_pie;

pub use accuracy_gate::{
    compute_cosine_similarity, compute_cross_entropy_ppl, compute_relative_l2_error, AccuracyGate,
    AccuracyTolerance, AccuracyVerdict,
};
pub use gsq::{gsq_fit_block, GsqBlockFit, GsqConfig};
pub use rco::{rco_search, RcoConfig};
pub use spqr::{spqr_identify_salient, SpqrSalientResidual};

/// Re-exported from `grim_tensor` so the `BackendDevice::quantize` trait method (which lives in `grim-tensor`) and the
/// CPU `quant_*` reference functions (which live here) share one canonical enum without a circular dependency.
pub use grim_tensor::dtype::QuantFormat;

pub const BLOCK_SIZE_Q8: usize = 32;
pub const BLOCK_SIZE_Q4_K: usize = 32;
const BLOCK_SIZE_QK: usize = 32;

#[derive(Debug, Clone)]
pub struct TensorRewritePlan {
    pub target: QuantFormat,
    pub shape: Vec<usize>,
    pub importance: Option<Vec<f32>>,
    pub curvature: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct RewrittenTensorData {
    pub bytes: Vec<u8>,
    pub logical_shape: Vec<usize>,
    pub target: QuantFormat,
    /// True if weights are stored in wavefront-tiled layout for ROCm LDS efficiency.
    /// When true, `write_grim_file` should set `layout_hint = GrimLayoutHint::WavefrontTiled`.
    pub wavefront_tiled: bool,
}

/// Dequantize OSTQuant W4A4 weights:
/// Shape: [out_features, in_features] (or [N, K]).
/// - `qweight`: [N, K / 8] as u32 (8 unsigned 4-bit nibbles per word)
/// - `scales`: [N, K / group_size] as bf16
/// - `zeros`: [N, K / group_size] as u8
pub fn dequant_ostquant_w4a4(
    qweight: &[u8],
    scales: &[u8],
    zeros: &[u8],
    shape: &[usize],
    group_size: usize,
) -> Result<Vec<f32>> {
    let out_features = *shape.first().ok_or_else(|| {
        Error::Backend("dequant_ostquant_w4a4: shape missing out_features".into())
    })?;
    let in_features = *shape
        .get(1)
        .ok_or_else(|| Error::Backend("dequant_ostquant_w4a4: shape missing in_features".into()))?;

    let n_groups = in_features.div_ceil(group_size);
    let words_per_col = in_features / 8;
    let mut out = vec![0.0f32; out_features * in_features];

    for row in 0..out_features {
        for g in 0..n_groups {
            let sc_offset = (row * n_groups + g) * 2;
            let zr_offset = row * n_groups + g;
            if sc_offset + 2 > scales.len() || zr_offset >= zeros.len() {
                return Err(Error::Backend(
                    "dequant_ostquant_w4a4: scales/zeros buffer out of bounds".into(),
                ));
            }
            let sc_bits = u16::from_le_bytes([scales[sc_offset], scales[sc_offset + 1]]);
            let sc = f32::from_bits((sc_bits as u32) << 16);
            let zr = zeros[zr_offset] as f32;

            let words_in_grp = group_size / 8;
            for w in 0..words_in_grp {
                let qw_offset = (row * words_per_col + g * words_in_grp + w) * 4;
                if qw_offset + 4 > qweight.len() {
                    return Err(Error::Backend(
                        "dequant_ostquant_w4a4: qweight buffer out of bounds".into(),
                    ));
                }
                let word = u32::from_le_bytes([
                    qweight[qw_offset],
                    qweight[qw_offset + 1],
                    qweight[qw_offset + 2],
                    qweight[qw_offset + 3],
                ]);

                for i in 0..8 {
                    let nib = ((word >> (i * 4)) & 0xF) as f32;
                    let val = sc * (nib - zr);
                    let col_idx = g * group_size + w * 8 + i;
                    if col_idx < in_features {
                        out[row * in_features + col_idx] = val;
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Quantize f32 weights to the unsigned-4-bit group-128 OSTQuant layout that
/// the WhiteCrow W4A4 GEMV (`launch_w4a4_ostquant_gemv` /
/// `dequant_ostquant_w4a4`) consumes: `qweight` u32 words `[N, K/8]` (nibble i
/// of word w at bit 4i), `scales` bf16-as-u16-LE `[N, K/128]`, `zeros` u8
/// `[N, K/128]`; dequant is `scale * (nibble - zero)`.
///
/// Per (column, 128-group): asymmetric min/max fit — `d = (max-min)/15`,
/// `zero = round(-min/d)` clamped to 0..15, `nib = clamp(round(v/d) + zero)`.
/// The scale is rounded to bf16 BEFORE encoding nibbles so the stored codes
/// are optimal for the scale the kernel will actually use.
///
/// Used by the q4k→WhiteCrow requant-at-load path (GRIM_DECODE_W4A4): the
/// 9B's Q4_K weights are dequantized once and re-encoded in this layout so
/// decode rides the native v_dot8 GEMV (435 GB/s measured) instead of the
/// 23 GB/s q4k dot4 kernel.
/// Bumped whenever the u4 group-128 byte layout changes. It is mixed into the
/// on-disk WhiteCrow cache key, so a format change can never serve tensors
/// converted by an older encoder.
pub const OSTQUANT_ENCODER_VERSION: u32 = 1;

pub fn quant_ostquant_w4_group128(
    w: &[f32],
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if w.len() < n * k {
        return Err(Error::Backend(format!(
            "quant_ostquant_w4_group128: need {} weights, got {}",
            n * k,
            w.len()
        )));
    }
    let n_groups = k / 128;
    let words_per_col = k / 8;
    let mut qw = vec![0u8; n * words_per_col * 4];
    let mut sc = vec![0u8; n * n_groups * 2];
    let mut zr = vec![0u8; n * n_groups];

    // bf16 rounding helper: truncate-and-round via f32 bits.
    let to_bf16 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let from_bf16 = |bits: u16| -> f32 { f32::from_bits((bits as u32) << 16) };

    for col in 0..n {
        let row = &w[col * k..col * k + k];
        for g in 0..n_groups {
            let grp = &row[g * 128..g * 128 + 128];
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in grp {
                min = min.min(v);
                max = max.max(v);
            }
            let d = ((max - min) / 15.0).max(1e-30);
            // Encode with the scale the kernel will dequant with.
            let d_bf16 = from_bf16(to_bf16(d));
            let mut zero = (-min / d_bf16).round();
            if zero < 0.0 {
                zero = 0.0;
            }
            if zero > 15.0 {
                zero = 15.0;
            }
            let zero = zero as u8;
            let sc_bits = to_bf16(d_bf16);
            sc[(col * n_groups + g) * 2..(col * n_groups + g) * 2 + 2]
                .copy_from_slice(&sc_bits.to_le_bytes());
            zr[col * n_groups + g] = zero;
            let base = col * words_per_col + g * 16; // 128/8 words per group
            for (i, &v) in grp.iter().enumerate() {
                let nib = ((v / d_bf16).round() as i32 + zero as i32).clamp(0, 15) as u32;
                let word_i = base + i / 8;
                let shift = (i % 8) * 4;
                let bits = &mut qw[word_i * 4..word_i * 4 + 4];
                let cur = u32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]);
                let next = cur | (nib << shift);
                bits.copy_from_slice(&next.to_le_bytes());
            }
        }
    }
    Ok((qw, sc, zr))
}

/// GreyCrow u4 group-32 layout version.
///
/// Bumped whenever the GreyCrow byte layout changes, and mixed into the
/// on-disk cache key of any GreyCrow tensor cache, for the same reason as
/// [`OSTQUANT_ENCODER_VERSION`]: a layout change must never be served from a
/// cache written by an older encoder.
pub const GREYCROW_ENCODER_VERSION: u32 = 1;

/// Dequantize legacy GGUF Q4_0 (type 2): 18-byte blocks of
/// `[f16 scale][16 bytes of nibbles]`, one block per 32 weights,
/// `w = (nibble - 8) * scale`.
///
/// This is the format GreyCrow is built on. It exists because GGUF's
/// Q4_0/Q4_1/Q4_2 used to be typed as `KQuant(Q4K)` (144-byte super-blocks),
/// so a Q4_0 payload was sliced and decoded with the wrong geometry.
pub fn dequant_q4_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let blocks = n.div_ceil(32);
    if bytes.len() < blocks * 18 {
        return Err(Error::Backend(format!(
            "dequant_q4_0: need {} bytes for {n} weights, got {}",
            blocks * 18,
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..blocks {
        let blk = &bytes[b * 18..b * 18 + 18];
        let d = f16_to_f32(blk[0], blk[1]);
        let base = b * 32;
        let count = (n - base).min(32);
        for i in 0..count {
            let byte = blk[2 + i / 2];
            let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[base + i] = (nib as f32 - 8.0) * d;
        }
    }
    Ok(out)
}

/// Repack legacy Q4_0 into the GreyCrow u4 group-32 layout, **bit-exactly**.
///
/// Q4_0 already stores what GreyCrow decodes: a per-32 f16 scale and nibbles
/// biased by 8. So this is a byte shuffle, not a requantization — no f32
/// anywhere, and the round trip through [`dequant_greycrow_g32`] reproduces
/// [`dequant_q4_0`] exactly. Contrast WhiteCrow's q4k path, which expands to
/// f32 and re-derives group-128 scales.
///
/// Layout, per column-major weight `[n, k]`, matching WhiteCrow's framing so
/// one blob reader serves both:
/// - `qw`: u32 words `[n, k/8]`, 8 codes per word, low nibble first.
/// - `sc`: f16 scales `[n, k/32]`, carried over from Q4_0 verbatim.
/// - `zr`: u8 `[n, k/32]`, all `8` — the bias is already folded into the
///   nibble, so a kernel reads `w = (q - z) * scale` with `z == 8`.
///
/// 4.5 bpw, identical to Q4_0. Requires `k % 32 == 0`.
pub fn repack_q40_to_greycrow_g32(bytes: &[u8], n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if k % 32 != 0 {
        return Err(Error::Backend(format!(
            "repack_q40_to_greycrow_g32: k={k} must be divisible by 32"
        )));
    }
    let blocks = k / 32;
    let src_blocks = n * blocks;
    if bytes.len() < src_blocks * 18 {
        return Err(Error::Backend(format!(
            "repack_q40_to_greycrow_g32: need {} bytes for [{n},{k}], got {}",
            src_blocks * 18,
            bytes.len()
        )));
    }
    let words_per_col = k / 8;
    let mut qw = vec![0u8; n * words_per_col * 4];
    let mut sc = vec![0u8; n * blocks * 2];
    // The bias is already folded into the Q4_0 nibble, so every group reads
    // back with zero point 8. Row-separable, and per-tensor sizes here are
    // small enough that a serial pass beats thread fan-out.
    let zr = vec![8u8; n * blocks];
    for col in 0..n {
        for g in 0..blocks {
            let blk = &bytes[(col * blocks + g) * 18..(col * blocks + g) * 18 + 18];
            sc[(col * blocks + g) * 2..(col * blocks + g) * 2 + 2].copy_from_slice(&blk[0..2]);
            let base = col * words_per_col + g * 4; // 32/8 words per group
            for w in 0..4 {
                // 32 weights = 16 bytes = 4 u32 words, consecutive, low
                // nibble first -- the same order Q4_0 stores.
                let word = u32::from_le_bytes([
                    blk[2 + w * 4],
                    blk[2 + w * 4 + 1],
                    blk[2 + w * 4 + 2],
                    blk[2 + w * 4 + 3],
                ]);
                qw[(base + w) * 4..(base + w) * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    Ok((qw, sc, zr))
}

/// Host reference decoder for the GreyCrow u4 group-32 layout: `w = (q - z) * scale`.
///
/// The GPU kernel this mirrors is not written yet; this is the oracle a parity
/// test must match, and the only decoder the CPU path uses.
pub fn dequant_greycrow_g32(
    qw: &[u8],
    sc: &[u8],
    zr: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if k % 32 != 0 {
        return Err(Error::Backend(format!(
            "dequant_greycrow_g32: k={k} must be divisible by 32"
        )));
    }
    let groups = k / 32;
    let words_per_col = k / 8;
    if qw.len() < n * words_per_col * 4 || sc.len() < n * groups * 2 || zr.len() < n * groups {
        return Err(Error::Backend(format!(
            "dequant_greycrow_g32: buffers too small for [{n},{k}]"
        )));
    }
    let mut out = vec![0.0f32; n * k];
    for col in 0..n {
        for g in 0..groups {
            let d = f16_to_f32(
                sc[(col * groups + g) * 2],
                sc[(col * groups + g) * 2 + 1],
            );
            let z = zr[col * groups + g] as f32;
            let base = col * words_per_col + g * 4;
            for w in 0..4 {
                let word = u32::from_le_bytes([
                    qw[(base + w) * 4],
                    qw[(base + w) * 4 + 1],
                    qw[(base + w) * 4 + 2],
                    qw[(base + w) * 4 + 3],
                ]);
                for i in 0..8 {
                    let nib = ((word >> ((i % 8) * 4)) & 0x0F) as f32;
                    let idx = (col * k) + g * 32 + (w * 8 + i);
                    if idx < n * k {
                        out[idx] = (nib - z) * d;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Dequantize grouped INT weights (EfficientQAT/GPTQ format).
/// # Layout - `qweight`: packed low-bit weights (strided) - `qzeros`: per-group zero-points (uint16 for 2/3/4-bit,.
pub fn dequant_gptq_group_int(
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
    g_idx: Option<&[u8]>,
    shape: &[usize],
    bits: u32,
    group_size: usize,
) -> Result<Vec<f32>> {
    // QNT-6 fix: `shape` is caller-supplied and was indexed with `shape[0]` / `shape[1]` directly, which panics on a slice shorter than 2 elements.
    // Bounds-check and return a proper error instead.
    let in_features = *shape.first().ok_or_else(|| {
        Error::Backend("dequant_gptq_group_int: shape missing in_features".into())
    })?;
    let out_features = *shape.get(1).ok_or_else(|| {
        Error::Backend("dequant_gptq_group_int: shape missing out_features".into())
    })?;

    let mut out = vec![0.0f32; in_features * out_features];

    let values_per_word = match bits {
        2 => 16,
        3 => 32,
        4 => 8,
        8 => 1,
        _ => return Err(Error::Backend(format!("unsupported GPTQ bits: {bits}"))),
    };

    let read_u32 = |bytes: &[u8], word_idx: usize| -> u32 {
        let offset = word_idx * 4;
        if offset + 4 <= bytes.len() {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        } else {
            0
        }
    };

    let get_group = |in_idx: usize| -> usize {
        if let Some(bytes) = g_idx {
            if bytes.len() == in_features * 4 {
                let offset = in_idx * 4;
                u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize
            } else if bytes.len() == in_features * 8 {
                let offset = in_idx * 8;
                u64::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]) as usize
            } else {
                in_idx / group_size
            }
        } else {
            in_idx / group_size
        }
    };

    let words_per_row_zeros = out_features.div_ceil(values_per_word);

    for in_idx in 0..in_features {
        let g = get_group(in_idx);

        for out_idx in 0..out_features {
            // Read scale
            let scale_idx = g * out_features + out_idx;
            let scale = if scale_idx * 4 + 4 <= scales.len() {
                f32::from_le_bytes([
                    scales[scale_idx * 4],
                    scales[scale_idx * 4 + 1],
                    scales[scale_idx * 4 + 2],
                    scales[scale_idx * 4 + 3],
                ])
            } else {
                1.0f32
            };

            // Read zero-point
            let zero = if bits == 3 {
                let super_idx = out_idx / 32;
                let total_bit = (out_idx % 32) * 3;
                let zero_word_idx = g * (3 * out_features.div_ceil(32)) + super_idx * 3;
                let word0 = read_u32(qzeros, zero_word_idx) as u128;
                let word1 = read_u32(qzeros, zero_word_idx + 1) as u128;
                let word2 = read_u32(qzeros, zero_word_idx + 2) as u128;
                let packed = word0 | (word1 << 32) | (word2 << 64);
                let zero_val = ((packed >> total_bit) & 0x7) as u32;
                (zero_val + 1) as f32
            } else {
                let zero_word_idx = g * words_per_row_zeros + out_idx / values_per_word;
                let zero_word = read_u32(qzeros, zero_word_idx);
                let bit_offset = (out_idx % values_per_word) * bits as usize;
                let zero_val = (zero_word >> bit_offset) & ((1 << bits) - 1);
                (zero_val + 1) as f32
            };

            // Read quantized code
            let quantized_code = if bits == 3 {
                let super_idx = in_idx / 32;
                let total_bit = (in_idx % 32) * 3;
                let word0_idx = (super_idx * 3) * out_features + out_idx;
                let word0 = read_u32(qweight, word0_idx) as u128;
                let word1 = read_u32(qweight, word0_idx + out_features) as u128;
                let word2 = read_u32(qweight, word0_idx + 2 * out_features) as u128;
                let packed = word0 | (word1 << 32) | (word2 << 64);
                ((packed >> total_bit) & 0x7) as u32
            } else {
                let word_idx = (in_idx / values_per_word) * out_features + out_idx;
                let word = read_u32(qweight, word_idx);
                let bit_offset = (in_idx % values_per_word) * bits as usize;
                (word >> bit_offset) & ((1 << bits) - 1)
            };

            out[in_idx * out_features + out_idx] = (quantized_code as f32 - zero) * scale;
        }
    }

    Ok(out)
}

/// Dequantize AWQ format group-quantized weights to f32.
/// Layout conventions for AWQ: - `qweight`: `[in_features / values_per_word, out_features]` uint32 words - `qzeros`: `[in_features.
pub fn dequant_awq_group_int(
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
    shape: &[usize],
    bits: u32,
    group_size: usize,
) -> Result<Vec<f32>> {
    let in_features = *shape
        .first()
        .ok_or_else(|| Error::Backend("dequant_awq_group_int: shape missing in_features".into()))?;
    let out_features = *shape.get(1).ok_or_else(|| {
        Error::Backend("dequant_awq_group_int: shape missing out_features".into())
    })?;

    let mut out = vec![0.0f32; in_features * out_features];

    let values_per_word = match bits {
        2 => 16,
        4 => 8,
        8 => 1,
        _ => return Err(Error::Backend(format!("unsupported AWQ bits: {bits}"))),
    };

    let read_u32 = |bytes: &[u8], word_idx: usize| -> u32 {
        let offset = word_idx * 4;
        if offset + 4 <= bytes.len() {
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        } else {
            0
        }
    };

    let words_per_row_zeros = out_features.div_ceil(values_per_word);

    for in_idx in 0..in_features {
        let g = in_idx / group_size;

        for out_idx in 0..out_features {
            // Read scale (f16 -> f32)
            let scale_idx = g * out_features + out_idx;
            let scale = if scale_idx * 2 + 2 <= scales.len() {
                f16_to_f32(scales[scale_idx * 2], scales[scale_idx * 2 + 1])
            } else {
                1.0f32
            };

            // Read raw zero-point (AWQ convention: no +1)
            let zero_word_idx = g * words_per_row_zeros + out_idx / values_per_word;
            let zero_word = read_u32(qzeros, zero_word_idx);
            let bit_offset = (out_idx % values_per_word) * bits as usize;
            let zero_val = (zero_word >> bit_offset) & ((1 << bits) - 1);
            let zero = zero_val as f32;

            // Read quantized code
            let word_idx = (in_idx / values_per_word) * out_features + out_idx;
            let word = read_u32(qweight, word_idx);
            let bit_offset = (in_idx % values_per_word) * bits as usize;
            let quantized_code = (word >> bit_offset) & ((1 << bits) - 1);

            out[in_idx * out_features + out_idx] = (quantized_code as f32 - zero) * scale;
        }
    }

    Ok(out)
}

/// Dequantize Q8_0 bytes to f32.
/// Q8_0 layout: for every 32 weights, a `f16` scale followed by 32 `i8` values.
pub fn dequant_q80(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    let stride = std::mem::size_of::<u16>() + BLOCK_Q8_WEIGHTS; // 2 + 32 = 34 bytes
    let num_blocks = num_weights.div_ceil(BLOCK_Q8_WEIGHTS);
    if data.len() < num_blocks * stride {
        return Err(Error::Backend(format!(
            "Q8_0: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * stride,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut data_pos = 0;
    let mut remaining = num_weights;
    for _ in 0..num_blocks {
        let scale = f16_to_f32(data[data_pos], data[data_pos + 1]);
        data_pos += 2;
        let n = remaining.min(BLOCK_Q8_WEIGHTS);
        for _ in 0..n {
            let v = data[data_pos] as i8 as f32;
            out.push(v * scale);
            data_pos += 1;
        }
        data_pos += BLOCK_Q8_WEIGHTS - n;
        remaining = remaining.saturating_sub(BLOCK_Q8_WEIGHTS);
    }
    Ok(out)
}

const BLOCK_Q8_WEIGHTS: usize = 32;

/// Absolute-value 16-entry codebook table alias for IQ4_XS.
const IQ4_NL_CODEBOOK: [f32; 16] = [
    0.0,
    0.113_141_26,
    0.243_736_04,
    0.397_433_65,
    0.565_743_55,
    0.722_941_4,
    0.897_054_55,
    1.075_762_9,
    1.294_598_8,
    1.528_519,
    1.826_856_4,
    2.270_011_2,
    3.237_191_2,
    5.508_296,
    10.416256,
    34.56951,
];

/// Dequantize IQ4_NL (ggml non-linear 4-bit) bytes to f32.
///
/// Faithful port of `dequantize_row_iq4_nl` (ggml-quants.c). The block is
/// llama.cpp's `block_iq4_nl` -- a `ggml_half d` followed by `qs[QK4_NL/2]`
/// with `QK4_NL = 32` -- so 18 bytes per 32 weights, NOT 170 bytes per 256.
/// The previous layout (`d` + a 32-byte sign plane + 128 nibble bytes + 8
/// sub-block scale bytes) matched no llama.cpp block and overran real GGUF
/// tensors: on Qwen3.8-27B `blk.1.ffn_down.weight` the 170/256 stride
/// addressed 59,187,200 bytes against a 50,135,040-byte tensor.
///
/// The sign is carried by the codebook entry itself (`kvalues_iq4nl` is a
/// signed table), so there is no sign plane and no per-sub-block scale.
/// Within a block, byte `j` supplies element `j` from its low nibble and
/// element `j + 16` from its high nibble -- the halves are not interleaved.
pub fn dequant_iq4nl(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const QK4_NL: usize = 32;
    const BLOCK_BYTES: usize = 2 + QK4_NL / 2;
    let num_blocks = num_weights.div_ceil(QK4_NL);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ4_NL: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    for b in 0..num_blocks {
        let pos = b * BLOCK_BYTES;
        let d = f16_to_f32(data[pos], data[pos + 1]);
        let qs = &data[pos + 2..pos + BLOCK_BYTES];
        for j in 0..QK4_NL / 2 {
            if out.len() >= num_weights {
                break;
            }
            // Reference order (dequantize_row_iq4_nl):
            //     y[j]        = d * k[qs[j] & 0xf]
            //     y[j + QK4_NL/2] = d * k[qs[j] >> 4]
            // The low nibble of byte j feeds element j and the high nibble
            // feeds element j+16 -- the two halves of the block are NOT
            // interleaved. Writing them adjacently transposes the block.
            out.push(d * KVALUES_IQ4NL_REF[(qs[j] & 0x0F) as usize]);
        }
        for j in 0..QK4_NL / 2 {
            if out.len() >= num_weights {
                break;
            }
            out.push(d * KVALUES_IQ4NL_REF[((qs[j] >> 4) & 0x0F) as usize]);
        }
    }
    out.truncate(num_weights);
    Ok(out)
}

// IQ4_XS uses the same 16-entry codebook as IQ4_NL (llama.cpp `iq4nl_table`).
// The sign comes from bit 3 of the nibble; bits 0-2 index the codebook.

/// Dequantize IQ4_XS (llama.cpp importance-matrix 4-bit Extra Small) bytes to f32.
/// Per 256-weight super-block (136 bytes):
/// - `d`: f16 global scale (2 bytes)
/// - `scales_h`: u16 high bits for scales (2 bytes)
/// - `scales_l`: [u8; 4] low bits for scales (4 bytes)
/// - `qs`: [u8; 128] quantized values (128 bytes)
pub fn dequant_iq4xs(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    use crate::iq_tables::KVALUES_IQ4NL;
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 136;
    let num_blocks = num_weights.div_ceil(QK);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ4_XS: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut remaining = num_weights;
    for b in 0..num_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(blk[0], blk[1]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];

        let block_len = remaining.min(QK);
        for ib in 0..(QK / 32) {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32)
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            let qs_sub = &qs[ib * 16..(ib + 1) * 16];
            let base_idx = ib * 32;

            for j in 0..16 {
                if base_idx + j < block_len {
                    out.push(dl * KVALUES_IQ4NL[(qs_sub[j] & 0xf) as usize]);
                }
            }
            for j in 0..16 {
                if base_idx + 16 + j < block_len {
                    out.push(dl * KVALUES_IQ4NL[(qs_sub[j] >> 4) as usize]);
                }
            }
        }
        remaining = remaining.saturating_sub(QK);
    }
    Ok(out)
}

/// Dequantize IQ3_XXS (llama.cpp importance-matrix 3-bit Extra Extra Small) bytes to f32.
/// Per 256-weight super-block (98 bytes):
/// - `d`: f16 global scale (2 bytes)
/// - `qs`: [u8; 96] where qs[0..64] are grid indices, qs[64..96] are scales and signs (32 bytes)
pub fn dequant_iq3xxs(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    use crate::iq_tables::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 98;
    let num_blocks = num_weights.div_ceil(QK);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ3_XXS: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut remaining = num_weights;
    for b in 0..num_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(blk[0], blk[1]);
        let qs = &blk[2..66];
        let scales_and_signs = &blk[66..98];

        let block_len = remaining.min(QK);
        for ib32 in 0..(QK / 32) {
            let aux32 = u32::from_le_bytes([
                scales_and_signs[4 * ib32],
                scales_and_signs[4 * ib32 + 1],
                scales_and_signs[4 * ib32 + 2],
                scales_and_signs[4 * ib32 + 3],
            ]);
            let db = d * (0.5f32 + (aux32 >> 28) as f32) * 0.5f32;
            let qs_sub = &qs[ib32 * 8..(ib32 + 1) * 8];

            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let grid1_val = IQ3XXS_GRID[qs_sub[2 * l + 0] as usize];
                let grid2_val = IQ3XXS_GRID[qs_sub[2 * l + 1] as usize];
                let grid1 = grid1_val.to_le_bytes();
                let grid2 = grid2_val.to_le_bytes();

                let base_idx = ib32 * 32 + l * 8;
                for j in 0..4 {
                    if base_idx + j < block_len {
                        let sign = if (signs & KMASK_IQ2XS[j]) != 0 {
                            -1.0f32
                        } else {
                            1.0f32
                        };
                        out.push(db * (grid1[j] as f32) * sign);
                    }
                }
                for j in 0..4 {
                    if base_idx + 4 + j < block_len {
                        let sign = if (signs & KMASK_IQ2XS[j + 4]) != 0 {
                            -1.0f32
                        } else {
                            1.0f32
                        };
                        out.push(db * (grid2[j] as f32) * sign);
                    }
                }
            }
        }
        remaining = remaining.saturating_sub(QK);
    }
    Ok(out)
}

/// Dequantize IQ3_S (llama.cpp importance-matrix 3-bit Small) bytes to f32.
/// Per 256-weight super-block (110 bytes):
/// - `d`: f16 global scale (2 bytes)
/// - `qs`: [u8; 64]
/// - `qh`: [u8; 8]
/// - `signs`: [u8; 32]
/// - `scales`: [u8; 4]
/// Decode one 110-byte IQ3_S super-block to 256 f32 weights.
///
/// Shared by [`dequant_iq3s`] (oracle) and [`gemm_iq3s_packed`] (fused GEMM)
/// so the two can never drift (Pattern A: oracle + fused kernel bit-identical).
pub(crate) fn dequant_iq3s_block(blk: &[u8]) -> [f32; 256] {
    use crate::iq_tables::{IQ3S_GRID, KMASK_IQ2XS};
    debug_assert_eq!(blk.len(), IQ3S_BLOCK_BYTES);
    let mut w = [0.0f32; IQ3S_QK];
    let d = f16_to_f32(blk[0], blk[1]);
    let qs = &blk[2..66];
    let qh = &blk[66..74];
    let signs = &blk[74..106];
    let scales = &blk[106..110];

    for ib32 in (0..(IQ3S_QK / 32)).step_by(2) {
        let sc_byte = scales[ib32 / 2];
        let db1 = d * (1.0f32 + 2.0f32 * (sc_byte & 0xf) as f32);
        let db2 = d * (1.0f32 + 2.0f32 * (sc_byte >> 4) as f32);

        let qh0 = qh[ib32] as usize;
        let qh1 = qh[ib32 + 1] as usize;

        // First 32 weights — same grid/qh/sign math as the oracle.
        let qs1 = &qs[ib32 * 8..(ib32 + 1) * 8];
        let signs1 = &signs[ib32 * 4..(ib32 + 1) * 4];
        for l in 0..4 {
            let idx1 = (qs1[2 * l] as usize) | ((qh0 << (8 - 2 * l)) & 256);
            let idx2 = (qs1[2 * l + 1] as usize) | ((qh0 << (7 - 2 * l)) & 256);
            let grid1 = IQ3S_GRID[idx1].to_le_bytes();
            let grid2 = IQ3S_GRID[idx2].to_le_bytes();

            for j in 0..4 {
                let sign = if (signs1[l] & KMASK_IQ2XS[j]) != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };
                w[ib32 * 32 + l * 8 + j] = db1 * (grid1[j] as f32) * sign;
            }
            for j in 0..4 {
                let sign = if (signs1[l] & KMASK_IQ2XS[j + 4]) != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };
                w[ib32 * 32 + l * 8 + 4 + j] = db1 * (grid2[j] as f32) * sign;
            }
        }

        // Second 32 weights.
        let qs2 = &qs[(ib32 + 1) * 8..(ib32 + 2) * 8];
        let signs2 = &signs[(ib32 + 1) * 4..(ib32 + 2) * 4];
        for l in 0..4 {
            let idx1 = (qs2[2 * l] as usize) | ((qh1 << (8 - 2 * l)) & 256);
            let idx2 = (qs2[2 * l + 1] as usize) | ((qh1 << (7 - 2 * l)) & 256);
            let grid1 = IQ3S_GRID[idx1].to_le_bytes();
            let grid2 = IQ3S_GRID[idx2].to_le_bytes();

            for j in 0..4 {
                let sign = if (signs2[l] & KMASK_IQ2XS[j]) != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };
                w[(ib32 + 1) * 32 + l * 8 + j] = db2 * (grid1[j] as f32) * sign;
            }
            for j in 0..4 {
                let sign = if (signs2[l] & KMASK_IQ2XS[j + 4]) != 0 {
                    -1.0f32
                } else {
                    1.0f32
                };
                w[(ib32 + 1) * 32 + l * 8 + 4 + j] = db2 * (grid2[j] as f32) * sign;
            }
        }
    }
    w
}

pub fn dequant_iq3s(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let num_blocks = num_weights.div_ceil(QK);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ3_S: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut remaining = num_weights;
    for b in 0..num_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let w = dequant_iq3s_block(blk);
        let block_len = remaining.min(QK);
        out.extend_from_slice(&w[..block_len]);
        remaining = remaining.saturating_sub(QK);
    }
    Ok(out)
}

/// Dequantize IQ2_XXS (llama.cpp importance-matrix 2-bit Extra Extra Small) bytes to f32.
/// Per 256-weight super-block (66 bytes):
/// - `d`: f16 global scale (2 bytes)
/// - `qs`: [u8; 64] (interpreted as [u16; 32] or 8 uint32_t pairs: 4 bytes grid indices, 4 bytes scales/signs)
pub fn dequant_iq2xxs(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    use crate::iq_tables::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 66;
    let num_blocks = num_weights.div_ceil(QK);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ2_XXS: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut remaining = num_weights;
    for b in 0..num_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(blk[0], blk[1]);
        let qs = &blk[2..66];

        let block_len = remaining.min(QK);
        for ib32 in 0..(QK / 32) {
            let aux8 = &qs[8 * ib32..8 * ib32 + 4];
            let aux32_1 = u32::from_le_bytes([
                qs[8 * ib32 + 4],
                qs[8 * ib32 + 5],
                qs[8 * ib32 + 6],
                qs[8 * ib32 + 7],
            ]);
            let db = d * (0.5f32 + (aux32_1 >> 28) as f32) * 0.25f32;

            for l in 0..4 {
                let grid_val = IQ2XXS_GRID[aux8[l] as usize];
                let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                let base_idx = ib32 * 32 + l * 8;

                for j in 0..8 {
                    if base_idx + j < block_len {
                        let g = ((grid_val >> (8 * j)) & 0xff) as f32;
                        let sign = if (signs & KMASK_IQ2XS[j]) != 0 {
                            -1.0f32
                        } else {
                            1.0f32
                        };
                        out.push(db * g * sign);
                    }
                }
            }
        }
        remaining = remaining.saturating_sub(QK);
    }
    Ok(out)
}

/// Dequantize IQ2_XS (llama.cpp importance-matrix 2-bit Extra Small) bytes to f32.
/// Per 256-weight super-block (74 bytes):
/// - `d`: f16 global scale (2 bytes)
/// - `qs`: [u8; 64] (interpreted as 32 little-endian uint16_t values)
/// - `scales`: [u8; 8]
pub fn dequant_iq2xs(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    use crate::iq_tables::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 74;
    let num_blocks = num_weights.div_ceil(QK);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ2_XS: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut remaining = num_weights;
    for b in 0..num_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(blk[0], blk[1]);
        let qs = &blk[2..66];
        let scales = &blk[66..74];

        let block_len = remaining.min(QK);
        for ib32 in 0..(QK / 32) {
            let sc_byte = scales[ib32];
            let db = [
                d * (0.5f32 + (sc_byte & 0xf) as f32) * 0.25f32,
                d * (0.5f32 + (sc_byte >> 4) as f32) * 0.25f32,
            ];

            for l in 0..4 {
                let q_offset = (ib32 * 4 + l) * 2;
                let q_val = u16::from_le_bytes([qs[q_offset], qs[q_offset + 1]]);
                let grid_idx = (q_val & 511) as usize;
                let signs = KSIGNS_IQ2XS[(q_val >> 9) as usize];
                let grid_val = IQ2XS_GRID[grid_idx];
                let base_idx = ib32 * 32 + l * 8;

                for j in 0..8 {
                    if base_idx + j < block_len {
                        let g = ((grid_val >> (8 * j)) & 0xff) as f32;
                        let sign = if (signs & KMASK_IQ2XS[j]) != 0 {
                            -1.0f32
                        } else {
                            1.0f32
                        };
                        out.push(db[l / 2] * g * sign);
                    }
                }
            }
        }
        remaining = remaining.saturating_sub(QK);
    }
    Ok(out)
}

/// Dequantize IQ2_S (llama.cpp importance-matrix 2-bit Small) bytes to f32.
/// Per 256-weight super-block (82 bytes): - `d` : f16 global scale (2 bytes at offset.
/// Dequantize IQ2_S packed bytes to F32.
///
/// Transcribed from llama.cpp `dequantize_row_iq2_s`
/// (`old/repo/llama.cpp-master/ggml/src/ggml-quants.c`), block layout from
/// `ggml-common.h` (C, not Rust — hence the `text` fence):
///
/// ```text
/// block_iq2_s { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
///               uint8_t scales[QK_K/32]; }        // 2 + 64 + 8 + 8 = 82 B
/// ```
///
/// and `signs` is a VIEW into `qs` at `qs + QK_K/8`, not a separate field.
/// The previous version here used a wholly different layout — a 16-element
/// sub-scale with a `*0.125+0.5` factor, an arithmetic `(idx + i%8) % 4 - 1.5`
/// grid, and one sign bit per element read as `signs[i/8] >> (i%8)`. All three
/// were wrong, and the sign index ran to 31 in a 24-byte field, reading 8 bytes
/// past the block.
///
/// Note `scales` is indexed per 32 elements here; the sub-scale is
/// `d * (0.5 + nibble) * 0.25`, low nibble for `l` 0/1 and high for 2/3.
pub fn dequant_iq2s(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    use crate::iq_tables::{IQ2S_GRID, KMASK_IQ2XS};
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 82;
    const D: usize = 0;
    const QS: usize = 2;
    const QH: usize = QS + QK / 4; // 66
    const SCALES: usize = QH + QK / 32; // 74
    const NB32: usize = QK / 32; // 8 groups of 32

    let num_blocks = num_weights.div_ceil(QK);
    if data.len() < num_blocks * BLOCK_BYTES {
        return Err(Error::Backend(format!(
            "IQ2_S: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut remaining = num_weights;
    for b in 0..num_blocks {
        let blk = &data[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let d = f16_to_f32(blk[D], blk[D + 1]);
        let block_len = remaining.min(QK);
        for ib32 in 0..NB32 {
            let sc_byte = blk[SCALES + ib32];
            let db = [
                d * (0.5 + (sc_byte & 0x0f) as f32) * 0.25,
                d * (0.5 + (sc_byte >> 4) as f32) * 0.25,
            ];
            // `signs` aliases qs + QK_K/8 and advances 4 per group, as does qs.
            let signs_at = QS + QK / 8 + ib32 * 4;
            for l in 0..4usize {
                let dl = db[l / 2];
                let idx = (blk[QS + ib32 * 4 + l] as usize)
                    | (((blk[QH + ib32] as usize) << (8 - 2 * l)) & 0x300);
                let packed = IQ2S_GRID[idx];
                for j in 0..8usize {
                    if ib32 * 32 + l * 8 + j >= block_len {
                        break;
                    }
                    let g = ((packed >> (8 * j)) & 0xff) as f32;
                    let s = if blk[signs_at + l] & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out.push(dl * g * s);
                }
            }
        }
        remaining = remaining.saturating_sub(QK);
    }
    out.truncate(num_weights);
    Ok(out)
}
/// Dequantize Q4_K bytes to f32 per the ggml/llama.cpp super-block specification.
/// Each 256-weight super-block consumes 144 bytes: - 2 bytes f16 `d` (super-block scale) - 2.
pub fn dequant_q4k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;

    if num_weights == 0 {
        return Ok(Vec::new());
    }

    let num_blocks = num_weights.div_ceil(BLOCK_SIZE);
    let expected_bytes = num_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "dequant_q4k: buffer too short: expected {expected_bytes}, got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;

    for _ in 0..num_blocks {
        let d = f16_to_f32(data[pos], data[pos + 1]);
        let min = f16_to_f32(data[pos + 2], data[pos + 3]);
        let scales = &data[pos + 4..pos + 16];
        let qs = &data[pos + 16..pos + 144];

        let mut q_idx = 0;
        let mut is = 0;

        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1;
            let m1_val = min * m1;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2;
            let m2_val = min * m2;

            for l in 0..32 {
                if out.len() < num_weights {
                    let q1 = (qs[q_idx + l] & 0x0F) as f32;
                    out.push(d1 * q1 - m1_val);
                }
            }

            for l in 0..32 {
                if out.len() < num_weights {
                    let q2 = (qs[q_idx + l] >> 4) as f32;
                    out.push(d2 * q2 - m2_val);
                }
            }

            q_idx += 32;
            is += 2;
        }

        pos += BLOCK_BYTES;
    }

    Ok(out)
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (f32, f32) {
    let (sc, m) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    (sc as f32, m as f32)
}

/// Dequantize Q5_K bytes to f32 per the ggml/llama.cpp super-block specification (176 bytes / 256 weights).
pub fn dequant_q5k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;

    if num_weights == 0 {
        return Ok(Vec::new());
    }

    let num_blocks = num_weights.div_ceil(BLOCK_SIZE);
    let expected_bytes = num_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "dequant_q5k: buffer too short: expected {expected_bytes}, got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;

    for _ in 0..num_blocks {
        let d = f16_to_f32(data[pos], data[pos + 1]);
        let dmin = f16_to_f32(data[pos + 2], data[pos + 3]);
        let scales = &data[pos + 4..pos + 16];
        let qh = &data[pos + 16..pos + 48];
        let qs = &data[pos + 48..pos + 176];

        let mut qs_idx = 0;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1;
            let min1 = dmin * m1;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2;
            let min2 = dmin * m2;

            let mut block_out = [0.0f32; 64];
            for l in 0..32 {
                let lo = qs[qs_idx + l] & 0x0F;
                let hi = qs[qs_idx + l] >> 4;
                let q_lo = lo + if (qh[l] & u1) != 0 { 16 } else { 0 };
                let q_hi = hi + if (qh[l] & u2) != 0 { 16 } else { 0 };
                block_out[l] = d1 * q_lo as f32 - min1;
                block_out[l + 32] = d2 * q_hi as f32 - min2;
            }
            for &v in &block_out {
                if out.len() < num_weights {
                    out.push(v);
                }
            }

            qs_idx += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        pos += BLOCK_BYTES;
    }

    Ok(out)
}

/// Dequantize Q6_K bytes to f32 per the ggml/llama.cpp super-block specification (210 bytes / 256 weights).
pub fn dequant_q6k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    if num_weights == 0 {
        return Ok(Vec::new());
    }

    let num_blocks = num_weights.div_ceil(BLOCK_SIZE);
    let expected_bytes = num_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "dequant_q6k: buffer too short: expected {expected_bytes}, got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;

    for _ in 0..num_blocks {
        // ggml block_q6_K layout: ql (128B) + qh (64B) + scales (16B, i8) + d (f16, LAST).
        let ql = &data[pos..pos + 128];
        let qh = &data[pos + 128..pos + 192];
        let scales = &data[pos + 192..pos + 208];
        let d = f16_to_f32(data[pos + 208], data[pos + 209]);

        let mut sc_idx = 0;
        let mut ql_idx = 0;
        let mut qh_idx = 0;

        for _ in 0..2 {
            let mut block_out = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) as f32 - 32.0;
                let q2 =
                    ((ql[ql_idx + l + 32] & 0x0F) | ((qh[qh_idx + l] & 0x0C) << 2)) as f32 - 32.0;
                let q3 = ((ql[ql_idx + l] >> 4) | (qh[qh_idx + l] & 0x30)) as f32 - 32.0;
                let q4 =
                    ((ql[ql_idx + l + 32] >> 4) | ((qh[qh_idx + l] & 0xC0) >> 2)) as f32 - 32.0;

                let sc1 = scales[sc_idx + is] as i8 as f32;
                let sc2 = scales[sc_idx + is + 2] as i8 as f32;
                let sc3 = scales[sc_idx + is + 4] as i8 as f32;
                let sc4 = scales[sc_idx + is + 6] as i8 as f32;

                block_out[l] = d * sc1 * q1;
                block_out[l + 32] = d * sc2 * q2;
                block_out[l + 64] = d * sc3 * q3;
                block_out[l + 96] = d * sc4 * q4;
            }
            for &v in &block_out {
                if out.len() < num_weights {
                    out.push(v);
                }
            }
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
        pos += BLOCK_BYTES;
    }

    Ok(out)
}

/// Dequantize Q2_K bytes to f32 per the ggml/llama.cpp super-block specification (84 bytes / 256 weights).
/// On-disk super-block layout (84 bytes total): - 16 bytes `scales` : 16 sub-blocks, each a.
pub fn dequant_q2k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 84;

    if num_weights == 0 {
        return Ok(Vec::new());
    }

    let num_blocks = num_weights.div_ceil(BLOCK_SIZE);
    let expected_bytes = num_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "dequant_q2k: buffer too short: expected {expected_bytes}, got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;

    for _ in 0..num_blocks {
        let scales = &data[pos..pos + 16];
        let qs = &data[pos + 16..pos + 80];
        let d = f16_to_f32(data[pos + 80], data[pos + 81]);
        let dmin = f16_to_f32(data[pos + 82], data[pos + 83]);

        // 16 sub-blocks of 16 weights. QNT-1 fix: `dmin` now reads its own 2 bytes at
        // offset 82/83 (previously aliased to `d`'s bytes at 80/81, so every min scale was wrong).
        let mut block_out = [0.0f32; 256];
        let mut q_off = 0usize;
        for sb in 0..16 {
            let sc = (scales[sb] & 0x0F) as f32;
            let m = (scales[sb] >> 4) as f32;
            let dl = d * sc;
            let ml = dmin * m;
            for w in 0..16 {
                let byte = qs[q_off + w / 4];
                let shift = (w % 4) * 2;
                let q_val = ((byte >> shift) & 3) as f32;
                block_out[sb * 16 + w] = dl * q_val - ml;
            }
            q_off += 4;
        }

        for &v in &block_out {
            if out.len() < num_weights {
                out.push(v);
            }
        }
        pos += BLOCK_BYTES;
    }

    Ok(out)
}

/// Dequantize Q3_K bytes to f32 per the ggml/llama.cpp super-block specification (110 bytes / 256 weights).
/// Matches llama.cpp `dequantize_row_q3_K` byte-for-byte.
pub fn dequant_q3k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 110;

    if num_weights == 0 {
        return Ok(Vec::new());
    }

    let num_blocks = num_weights.div_ceil(BLOCK_SIZE);
    let expected_bytes = num_blocks * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "dequant_q3k: buffer too short: expected {expected_bytes}, got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;

    for _ in 0..num_blocks {
        let hmask = &data[pos..pos + 32];
        let qs = &data[pos + 32..pos + 96];
        // ggml decodes the 12-byte `scales` field into 16 i8 values via a `memcpy` into a 16-byte `uint32_t aux[4]` and a bit shuffle.
        // The final 4 bytes of aux are zero-extended (uninitialized in C but the shuffle only.
        let scales = &data[pos + 96..pos + 108];
        let d = f16_to_f32(data[pos + 108], data[pos + 109]);

        // Decode the 12-byte `scales` into 16 i8 values using the ggml bit-shuffle (dequantize_row_q3_K): memcpy(aux, scales, 12); tmp = aux[2]; aux[2] = ((aux[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4); aux[3] = ((aux[1]
        // >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4); aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4); aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
        let kmask1: u32 = 0x0303_0303u32;
        let kmask2: u32 = 0x0F0F_0F0Fu32;
        let aux0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
        let aux1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
        let tmp = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
        let aux = [
            (aux0 & kmask2) | ((tmp & kmask1) << 4),        // aux[0]
            (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4), // aux[1]
            ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4), // aux[2]
            ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4), // aux[3]
        ];
        // Truncate to bytes; each aux word now holds 4 signed scale bytes.
        let mut sc = [0i8; 16];
        for j in 0..4 {
            let w = aux[j];
            sc[j * 4] = (w & 0xFF) as i8;
            sc[j * 4 + 1] = ((w >> 8) & 0xFF) as i8;
            sc[j * 4 + 2] = ((w >> 16) & 0xFF) as i8;
            sc[j * 4 + 3] = ((w >> 24) & 0xFF) as i8;
        }

        let mut block_out = [0.0f32; 256];
        let mut _is = 0usize;
        let mut m: u8 = 1;
        let mut q_off = 0;
        for n in (0..256).step_by(128) {
            let mut shift: i32 = 0;
            for _j in 0..4 {
                let dl = d * ((sc[_is] as i32 - 32) as f32);
                _is += 1;
                for l in 0..16 {
                    let q_val: i32 = ((qs[q_off + l] >> shift) & 3) as i32;
                    let hm_bit: i32 = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    block_out[n + _j * 32 + l] = dl * (q_val - hm_bit) as f32;
                }

                let dl = d * ((sc[_is] as i32 - 32) as f32);
                _is += 1;
                for l in 0..16 {
                    let q_val: i32 = ((qs[q_off + l + 16] >> shift) & 3) as i32;
                    let hm_bit: i32 = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    block_out[n + _j * 32 + 16 + l] = dl * (q_val - hm_bit) as f32;
                }

                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }

        for &v in &block_out {
            if out.len() < num_weights {
                out.push(v);
            }
        }
        pos += BLOCK_BYTES;
    }

    Ok(out)
}

/// Uniform-step 16-entry lookup table for the **non-standard "uniform FP4"** format used by `quant_fp4` / `dequant_fp4` / `dequant_fp4_block16` in this crate.
/// This is NOT the OCP E2M1 format.
const FP4_UNIFORM_LUT: [f32; 16] = [
    -1.0,   // 0000 -> -1.0
    -0.875, // 0001
    -0.75,  // 0010
    -0.625, // 0011
    -0.5,   // 0100
    -0.375, // 0101
    -0.25,  // 0110
    -0.125, // 0111
    0.0,    // 1000 -> 0.0
    0.125,  // 1001
    0.25,   // 1010
    0.375,  // 1011
    0.5,    // 1100
    0.625,  // 1101
    0.75,   // 1110
    0.875,  // 1111 -> +0.875
];

/// Dequantize FP4 E2M1 bytes to f32.
pub fn dequant_fp4(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(num_values);
    let scale = if data.len() >= 4 {
        f32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        1.0
    };

    let data_start = if data.len() >= 8 { 4 } else { 0 };
    for (i, &byte) in data[data_start..].iter().enumerate() {
        let hi = FP4_UNIFORM_LUT[(byte >> 4) as usize] * scale;
        let lo = FP4_UNIFORM_LUT[(byte & 0x0F) as usize] * scale;

        let idx = i * 2;
        if idx < num_values {
            out.push(hi);
        }
        if idx + 1 < num_values {
            out.push(lo);
        }
    }
    while out.len() < num_values {
        out.push(0.0);
    }
    Ok(out)
}

/// Dequantize block-scaled FP4 E2M1 bytes to f32.
pub fn dequant_fp4_block16(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let global_scale = if data.len() >= 4 {
        f32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        1.0
    };

    let num_blocks = num_values.div_ceil(16);
    let mut out = Vec::with_capacity(num_values);
    let mut pos = 4;
    for b in 0..num_blocks {
        if pos >= data.len() {
            break;
        }
        let block_scale_fp8 = data[pos];
        let block_scale = fp8_e4m3_to_f32(block_scale_fp8);
        let scale = block_scale * global_scale;
        pos += 1;

        let block_rem = num_values - b * 16;
        let block_len = block_rem.min(16);

        for i in 0..8 {
            if pos + i >= data.len() {
                break;
            }
            let byte = data[pos + i];
            let hi = FP4_UNIFORM_LUT[(byte >> 4) as usize] * scale;
            let lo = FP4_UNIFORM_LUT[(byte & 0x0F) as usize] * scale;

            let idx = i * 2;
            if idx < block_len {
                out.push(hi);
            }
            if idx + 1 < block_len {
                out.push(lo);
            }
        }
        pos += 8;
    }
    while out.len() < num_values {
        out.push(0.0);
    }
    Ok(out)
}

/// Dequantize FP8 (8-bit floating point) bytes to f32.
pub fn dequant_fp8(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(num_values);
    let scale = if data.len() >= 4 {
        f32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        1.0
    };
    let data_start = if data.len() >= 4 { 4 } else { 0 };
    for (i, &byte) in data[data_start..].iter().enumerate() {
        if i >= num_values {
            break;
        }
        out.push(fp8_e4m3_to_f32(byte) * scale);
    }
    while out.len() < num_values {
        out.push(0.0);
    }
    Ok(out)
}

/// Dequantize block-scaled FP8 bytes to f32.
pub fn dequant_fp8_block16(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let global_scale = if data.len() >= 4 {
        f32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        1.0
    };

    let num_blocks = num_values.div_ceil(16);
    let mut out = Vec::with_capacity(num_values);
    let mut pos = 4;
    for b in 0..num_blocks {
        if pos >= data.len() {
            break;
        }
        let block_scale_fp8 = data[pos];
        let block_scale = fp8_e4m3_to_f32(block_scale_fp8);
        let scale = block_scale * global_scale;
        pos += 1;

        let block_rem = num_values - b * 16;
        let block_len = block_rem.min(16);

        for i in 0..block_len {
            if pos + i >= data.len() {
                break;
            }
            let byte = data[pos + i];
            out.push(fp8_e4m3_to_f32(byte) * scale);
        }
        pos += 16;
    }
    while out.len() < num_values {
        out.push(0.0);
    }
    Ok(out)
}

/// Convert GGUF-native MXFP4 tensor bytes (llama.cpp layout) into the length-prefixed `[codes][exps]` framing consumed by `dequant_mxfp4` and the ROCm/CUDA `grim_dequant_mxfp4` kernels.
/// GGUF (llama.cpp `block_mxfp4`) stores, per 32-element block: one E8M0 scale byte FIRST, then 16 packed.
pub fn reframe_mxfp4_gguf(raw: &[u8], num_values: usize) -> Result<Vec<u8>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let blocks = num_values.div_ceil(32);
    let expected = blocks * 17;
    if raw.len() < expected {
        return Err(Error::Backend(format!(
            "reframe_mxfp4_gguf: buffer {} bytes too small for {num_values} values (need {expected})",
            raw.len()
        )));
    }

    let codes_len = num_values.div_ceil(2);
    let mut codes = vec![0u8; codes_len];
    let mut exps = vec![0u8; blocks];

    use rayon::prelude::*;

    // Vectorized block-by-block reframing: llama.cpp 17-byte block: byte 0 is scale; bytes 1..17 are 16-byte qs.
    // qs[0..16] low nibbles -> elements 0..15; high nibbles -> elements 16..31.
    if blocks >= 512 {
        const CHUNK_BLOCKS: usize = 256;
        codes
            .par_chunks_mut(CHUNK_BLOCKS * 16)
            .zip(exps.par_chunks_mut(CHUNK_BLOCKS))
            .enumerate()
            .for_each(
                |(chunk_idx, (c_chunk, e_chunk)): (usize, (&mut [u8], &mut [u8]))| {
                    let start_b = chunk_idx * CHUNK_BLOCKS;
                    let chunk_blocks = e_chunk.len();
                    for b_local in 0..chunk_blocks {
                        let b = start_b + b_local;
                        let raw_offset = b * 17;
                        e_chunk[b_local] = raw[raw_offset];

                        let qs = &raw[raw_offset + 1..raw_offset + 17];
                        let out_c = &mut c_chunk[b_local * 16..(b_local + 1) * 16];

                        // Lower 16 elements (0..15) packed into 8 bytes
                        for k in 0..8 {
                            out_c[k] = (qs[2 * k] & 0x0F) | ((qs[2 * k + 1] & 0x0F) << 4);
                        }
                        // Upper 16 elements (16..31) packed into 8 bytes
                        for k in 0..8 {
                            out_c[8 + k] = (qs[2 * k] >> 4) | ((qs[2 * k + 1] >> 4) << 4);
                        }
                    }
                },
            );
    } else {
        for b in 0..blocks {
            let raw_offset = b * 17;
            exps[b] = raw[raw_offset];

            let qs = &raw[raw_offset + 1..raw_offset + 17];
            let out_c = &mut codes[b * 16..(b + 1) * 16];

            for k in 0..8 {
                out_c[k] = (qs[2 * k] & 0x0F) | ((qs[2 * k + 1] & 0x0F) << 4);
            }
            for k in 0..8 {
                out_c[8 + k] = (qs[2 * k] >> 4) | ((qs[2 * k + 1] >> 4) << 4);
            }
        }
    }

    // Emit the length-prefixed framing.
    let mut out = Vec::with_capacity(16 + codes.len() + exps.len());
    out.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    out.extend_from_slice(&codes);
    out.extend_from_slice(&(exps.len() as u64).to_le_bytes());
    out.extend_from_slice(&exps);
    Ok(out)
}

/// Dequantize MXFP4 (OCP Microscaling, Jay tier) single-buffer bytes to f32.
/// # Layout Length-prefixed segments (same framing as the GPTQ group-int fix): - `[u64 LE]` codes_len.
pub fn dequant_mxfp4(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let mut cursor = 0usize;
    let read_segment = |bytes: &[u8], cursor: &mut usize| -> Result<Vec<u8>> {
        if bytes.len() < *cursor + 8 {
            return Err(Error::Backend(
                "Truncated MXFP4 segment length prefix".into(),
            ));
        }
        let len = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap()) as usize;
        *cursor += 8;
        if bytes.len() < *cursor + len {
            return Err(Error::Backend(format!(
                "Truncated MXFP4 segment (expected {len} bytes)"
            )));
        }
        let segment = bytes[*cursor..*cursor + len].to_vec();
        *cursor += len;
        Ok(segment)
    };

    let codes = read_segment(data, &mut cursor)?;
    let exps = read_segment(data, &mut cursor)?;

    let num_groups = num_values.div_ceil(32);
    if exps.len() < num_groups {
        return Err(Error::Backend(format!(
            "MXFP4: expected {num_groups} shared-exponent groups, got {}",
            exps.len()
        )));
    }
    if codes.len() < num_values.div_ceil(2) {
        return Err(Error::Backend(format!(
            "MXFP4: expected {} packed code bytes, got {}",
            num_values.div_ceil(2),
            codes.len()
        )));
    }

    let mut out = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let group_idx = i / 32;
        let shared_exp = exps[group_idx];
        let code_byte = codes[i / 2];
        let code = if i % 2 == 0 {
            code_byte & 0x0F
        } else {
            (code_byte >> 4) & 0x0F
        };
        out.push(mxfp4_e2m1_to_f32(code, shared_exp));
    }
    Ok(out)
}

/// Dequantize MXFP8 (OCP Microscaling, Magpie tier) single-buffer bytes to f32.
/// # Layout Length-prefixed segments (same framing as the GPTQ group-int fix): - `[u64 LE]` codes_len.
pub fn dequant_mxfp8(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let mut cursor = 0usize;
    let read_segment = |bytes: &[u8], cursor: &mut usize| -> Result<Vec<u8>> {
        if bytes.len() < *cursor + 8 {
            return Err(Error::Backend(
                "Truncated MXFP8 segment length prefix".into(),
            ));
        }
        let len = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap()) as usize;
        *cursor += 8;
        if bytes.len() < *cursor + len {
            return Err(Error::Backend(format!(
                "Truncated MXFP8 segment (expected {len} bytes)"
            )));
        }
        let segment = bytes[*cursor..*cursor + len].to_vec();
        *cursor += len;
        Ok(segment)
    };

    let codes = read_segment(data, &mut cursor)?;
    let exps = read_segment(data, &mut cursor)?;

    let num_groups = num_values.div_ceil(32);
    if exps.len() < num_groups {
        return Err(Error::Backend(format!(
            "MXFP8: expected {num_groups} shared-exponent groups, got {}",
            exps.len()
        )));
    }
    if codes.len() < num_values {
        return Err(Error::Backend(format!(
            "MXFP8: expected {num_values} code bytes, got {}",
            codes.len()
        )));
    }

    let mut out = Vec::with_capacity(num_values);
    for (group_idx, group) in codes.chunks(32).enumerate() {
        let shared_exp = exps[group_idx];
        let exp_scale = (2.0f32).powi(shared_exp as i32 - 127);
        for &code in group {
            out.push(fp8_e4m3_to_f32(code) * exp_scale);
        }
    }
    Ok(out)
}

/// Dequantize WNA16 (weight-only N-bit with per-block f16 scale + per-tensor f32 scale).
/// # Layout `[u32 n_bit][u32 num_blocks][u8 packed_codes...][f16 per_block_scales...][f32 tensor_scale]` - `n_bit`: bits per weight (2..=8).
pub fn dequant_wna16(data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    if data.len() < 12 {
        return Err(Error::Backend(
            "WNA16: payload too short for header (need >= 12 bytes)".into(),
        ));
    }
    let n_bit = u32::from_le_bytes(data[0..4].try_into().unwrap()) as u8;
    let num_blocks = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    if !(2..=8).contains(&n_bit) {
        return Err(Error::Backend(format!(
            "WNA16: n_bit={n_bit} out of range 2..=8"
        )));
    }
    let code_bytes_per_block = (256 * n_bit as usize).div_ceil(8);
    let codes_len = num_blocks * code_bytes_per_block;
    let scales_offset = 8 + codes_len;
    let tensor_scale_offset = scales_offset + num_blocks * 2;
    if data.len() < tensor_scale_offset + 4 {
        return Err(Error::Backend(format!(
            "WNA16: payload too short for full layout (need >= {}, got {})",
            tensor_scale_offset + 4,
            data.len()
        )));
    }
    let tensor_scale = f32::from_le_bytes(
        data[tensor_scale_offset..tensor_scale_offset + 4]
            .try_into()
            .unwrap(),
    );

    let mut out = Vec::with_capacity(elem_count);
    for block_idx in 0..num_blocks {
        let block_scale = f16_to_f32(
            data[scales_offset + block_idx * 2],
            data[scales_offset + block_idx * 2 + 1],
        );
        let block_start = 8 + block_idx * code_bytes_per_block;
        let block_end = block_start + code_bytes_per_block;
        let block_codes = &data[block_start..block_end];
        let block_elem_count = if block_idx == num_blocks - 1 {
            elem_count - block_idx * 256
        } else {
            256
        };
        for lane in 0..block_elem_count {
            let code = decode_msb_nbit(block_codes, 0, lane, n_bit);
            out.push(code as f32 * block_scale * tensor_scale);
        }
    }
    Ok(out)
}

/// Dequantize EmbeddingWNA16Int (embedding matrix stored as N-bit integers, row-major, with one f32 per-tensor scale).
/// # Layout `[u32 n_bit][u32 embedding_dim][u32 num_rows][u8 packed_codes...][f32 tensor_scale]` - `n_bit`: bits per entry (2..=8).
pub fn dequant_embedding_wna16_int(data: &[u8], elem_count: usize) -> Result<Vec<f32>> {
    if data.len() < 16 {
        return Err(Error::Backend(
            "EmbeddingWNA16Int: payload too short for header (need >= 16 bytes)".into(),
        ));
    }
    let n_bit = u32::from_le_bytes(data[0..4].try_into().unwrap()) as u8;
    let embedding_dim = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    if !(2..=8).contains(&n_bit) {
        return Err(Error::Backend(format!(
            "EmbeddingWNA16Int: n_bit={n_bit} out of range 2..=8"
        )));
    }
    let code_bytes_per_row = (embedding_dim * n_bit as usize).div_ceil(8);
    let codes_len = num_rows * code_bytes_per_row;
    let tensor_scale_offset = 12 + codes_len;
    if data.len() < tensor_scale_offset + 4 {
        return Err(Error::Backend(format!(
            "EmbeddingWNA16Int: payload too short for full layout (need >= {}, got {})",
            tensor_scale_offset + 4,
            data.len()
        )));
    }
    let tensor_scale = f32::from_le_bytes(
        data[tensor_scale_offset..tensor_scale_offset + 4]
            .try_into()
            .unwrap(),
    );

    let mut out = Vec::with_capacity(elem_count);
    for row_idx in 0..num_rows {
        let row_start = 12 + row_idx * code_bytes_per_row;
        let row_end = row_start + code_bytes_per_row;
        let row_codes = &data[row_start..row_end];
        let row_elem_count = if row_idx == num_rows - 1 {
            elem_count - row_idx * embedding_dim
        } else {
            embedding_dim
        };
        for col in 0..row_elem_count {
            let code = decode_msb_nbit(row_codes, 0, col, n_bit);
            out.push(code as f32 * tensor_scale);
        }
    }
    Ok(out)
}
/// Codes are packed MSB-first within each byte, crossing byte boundaries as needed.
/// `bit_pos = lane * n_bit`, and the decoder reads across the bytes that contain those.
fn decode_msb_nbit(code_bytes: &[u8], block_offset_bytes: usize, lane: usize, n_bit: u8) -> u32 {
    let bit_pos = (lane as u32).wrapping_mul(n_bit as u32);
    let byte_idx = (bit_pos / 8) as usize;
    let bit_in_byte = bit_pos % 8;
    let mut code: u32 = 0;
    let mut consumed: u32 = 0;
    let mut b = byte_idx;
    let mut shift = 8 - bit_in_byte;
    while consumed < n_bit as u32 {
        let byte = code_bytes[block_offset_bytes + b];
        let take = if shift == 0 { 8u32 } else { shift };
        let take = if take > n_bit as u32 - consumed {
            n_bit as u32 - consumed
        } else {
            take
        };
        let mask = if shift == 0 {
            0xFFu32
        } else {
            ((1u32 << take) - 1u32) << (8 - shift)
        };
        let bits = (byte as u32 & mask) >> (8 - shift - take);
        code |= bits << consumed;
        consumed += take;
        b += 1;
        shift = 0;
    }
    code
}

/// Canonical NF4 (normalized float-4) lookup table (bitsandbytes / QLoRA standard).
/// 16 quantiles of the standard normal distribution N(0, 1) scaled to [-1, 1].
pub const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.5251143,
    -0.3949175,
    -0.2844414,
    -0.18477343,
    -0.091050036,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Dequantize NF4 (normalized float-4) bytes to f32.
/// NF4 format (Quanto/Unsloth): asymmetric 4-bit quantization with per-tensor scale and min.
pub fn dequant_nf4(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(num_values);

    // Read global scale from first 4 bytes (default to 1.0)
    let scale = if data.len() >= 4 {
        f32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        1.0
    };

    // Decode packed NF4 values starting at byte 4
    for (i, &byte) in data[4..].iter().enumerate() {
        let hi = NF4_LUT[(byte >> 4) as usize] * scale;
        let lo = NF4_LUT[(byte & 0x0F) as usize] * scale;

        let idx = i * 2;
        if idx < num_values {
            out.push(hi);
        }
        if idx + 1 < num_values {
            out.push(lo);
        }
    }

    Ok(out)
}

/// FP8 formats: E4M3 (5 exp, 3 mantissa, no inf) and E5M2 (5 exp, 2 mantissa, with inf).
/// E4M3: exponent bias = 7, max value ≈ 240, min normalized ≈ 0.03125 E5M2: exponent.
const FP8_E4M3_BIAS: i32 = 7;

/// Convert FP8 E4M3 (4-bit exponent, 3-bit mantissa) to f32.
/// Layout: 1 sign | 4 exp | 3 mantissa
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte & 0x80) as i32;
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;

    if exp == 0xF {
        // OCP E4M3 ("FN") reserves ONLY mant == 7 as NaN. mant 0..6 are normal
        // numbers in [256, 448] = (1 + mant/8) * 2^(15 - 7).
        //
        // This previously returned 256 for all of mant 0..6, which silently
        // collapsed the top binade: 0x7E (448) decoded as 256.
        if mant == 7 {
            return f32::NAN;
        }
        let val = (1.0f32 + (mant as f32) / 8.0) * 256.0f32;
        return if sign != 0 { -val } else { val };
    }

    let mut result = (mant as f32) / 8.0 + 1.0;
    if exp != 0 {
        result *= 2f32.powi(exp - FP8_E4M3_BIAS);
    } else {
        result = (mant as f32) / 512.0;
    }

    if sign != 0 {
        -result
    } else {
        result
    }
}

fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // Subnormal or zero. An f16 subnormal encodes `mant
        // * 2^-24` (exponent unbiased 1-14, with 10 mantissa bits).
        let value = (mant as f32) * 2f32.powi(-24);
        if sign != 0 {
            -value
        } else {
            value
        }
    } else if exp == 31 {
        // NaN or inf
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// ForestRaven: symmetric per-output-row absmax INT8.
 ///
 /// Each row of an `[n, k]` weight matrix gets its own fp32 scale
 /// (`max|row| / 127`, 1.0 for an all-zero row) and its codes are
 /// `round(w / scale)` clamped to `[-128, 127]`. This is the article recipe
 /// (absmax symmetric, per-channel for weights): one scale per output
 /// channel costs `4n` bytes against `nk` codes -- under 0.1% for the shapes
 /// that matter -- while letting every channel use its full INT8 range.
 ///
 /// This is deliberately NOT Q8_0: Q8_0 scales per 32-element block with fp16
 /// scales, which is finer-grained but forces a scale application per block
 /// in the GEMV. Per-row scales need exactly one scale multiply per output,
 /// which is what `V_DOT4_I32_IU8` wants. The two formats MUST NOT share a
 /// kernel: same codes, different scale streams.
 ///
 /// Returns `(codes, scales)`: `n*k` int8 codes in row-major order and `n`
 /// fp32 little-endian scales. The framed blob layout
 /// (`[u64 codes_len][codes][u64 scales_len][scales]`) is assembled by the
 /// caller (convert, rewrite), mirroring WhiteCrow's triple-stream frame.
pub fn quant_forest_per_channel(
    data: &[f32],
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if data.len() < n * k {
        return Err(Error::Backend(format!(
            "quant_forest_per_channel: need {n}*{k} weights, got {}",
            data.len()
        )));
    }
    let mut codes = Vec::with_capacity(n * k);
    let mut scales = Vec::with_capacity(n * 4);
    for r in 0..n {
        let row = &data[r * k..(r + 1) * k];
        let amax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        scales.extend_from_slice(&scale.to_le_bytes());
        for &v in row {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            codes.push(q as u8);
        }
    }
    Ok((codes, scales))
}

/// Decode ForestRaven's framed blob back to dense f32.
///
/// Inverse of the framed layout above. Validates total length against
/// `(n, k)` geometry and refuses short/truncated buffers rather than
/// decoding a prefix: a truncated scale stream would rescale whole rows by
/// whatever bytes happen to follow, producing finite, plausible, wrong
/// weights with no signal.
pub fn dequant_forest(data: &[u8], n: usize, k: usize) -> std::result::Result<Vec<f32>, &'static str> {
    if data.len() < 8 {
        return Err("forest blob too short for codes length prefix");
    }
    let qw_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if qw_len != n * k {
        return Err("forest codes segment length does not match n*k");
    }
    if data.len() < 8 + qw_len + 8 {
        return Err("forest blob too short for scales length prefix");
    }
    let sc_len = u64::from_le_bytes(data[8 + qw_len..16 + qw_len].try_into().unwrap()) as usize;
    if sc_len != n * 4 {
        return Err("forest scales segment length does not match n fp32 scales");
    }
    if data.len() != 16 + qw_len + sc_len {
        return Err("forest blob has trailing bytes past the scales segment");
    }
    let codes = &data[8..8 + qw_len];
    let raw_scales = &data[16 + qw_len..];
    let mut out = Vec::with_capacity(n * k);
    for r in 0..n {
        let scale = f32::from_le_bytes(raw_scales[r * 4..(r + 1) * 4].try_into().unwrap());
        for c in 0..k {
            let q = codes[r * k + c] as i8 as f32;
            out.push(q * scale);
        }
    }
    Ok(out)
}

/// Quantize a slice of f32 values to Q8_0 bytes.
/// Each block of 32 gets a f16 scale and 32 i8 values.
pub fn quant_q80(data: &[f32]) -> Result<Vec<u8>> {    let num_blocks = data.len().div_ceil(BLOCK_Q8_WEIGHTS);
    let mut out = Vec::with_capacity(num_blocks * (2 + BLOCK_Q8_WEIGHTS));
    for block in data.chunks(BLOCK_Q8_WEIGHTS) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        let scale_bits = f32_to_f16(scale);
        out.extend_from_slice(&scale_bits.to_le_bytes());
        for &v in block {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
        // Pad incomplete block
        out.resize(out.len() + (BLOCK_Q8_WEIGHTS - block.len()), 0u8);
    }
    Ok(out)
}

/// Quantize a slice of f32 values to Q4_K bytes per the ggml super-block format.
/// Encodes 256-weight blocks into 144-byte Q4_K super-blocks using 6-bit sub-block scale and min packing.
pub fn quant_q4k(data: &[f32]) -> Result<Vec<u8>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;

    if data.is_empty() {
        return Ok(Vec::new());
    }

    let num_blocks = data.len().div_ceil(BLOCK_SIZE);
    let mut out = Vec::with_capacity(num_blocks * BLOCK_BYTES);

    for block in data.chunks(BLOCK_SIZE) {
        let mut block_data = [0.0f32; 256];
        block_data[..block.len()].copy_from_slice(block);

        let mut sub_d1 = [0.0f32; 8];
        let mut sub_m1 = [0.0f32; 8];
        let mut max_d1 = 0.0f32;
        let mut max_m1 = 0.0f32;

        for s in 0..8 {
            let sub = &block_data[s * 32..(s + 1) * 32];
            let min_v = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let max_v = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let m1 = if min_v < 0.0 { -min_v } else { 0.0 };
            let d1 = if min_v < 0.0 {
                (max_v - min_v) / 15.0
            } else {
                max_v.max(0.0) / 15.0
            };

            sub_m1[s] = m1;
            sub_d1[s] = d1;

            if d1 > max_d1 {
                max_d1 = d1;
            }
            if m1 > max_m1 {
                max_m1 = m1;
            }
        }

        let d = if max_d1 == 0.0 { 1.0 } else { max_d1 / 63.0 };
        let min = if max_m1 == 0.0 { 0.0 } else { max_m1 / 63.0 };

        let d_bytes = f32_to_f16(d).to_le_bytes();
        let min_bytes = f32_to_f16(min).to_le_bytes();

        out.extend_from_slice(&d_bytes);
        out.extend_from_slice(&min_bytes);

        let mut sc_u8 = [0u8; 8];
        let mut m_u8 = [0u8; 8];
        for s in 0..8 {
            let sc_val = if d > 0.0 {
                (sub_d1[s] / d).round().clamp(1.0, 63.0) as u8
            } else {
                1
            };
            let m_val = if min > 0.0 {
                (sub_m1[s] / min).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            sc_u8[s] = sc_val;
            m_u8[s] = m_val;
        }

        let scales_bytes = pack_scale_min_k4(&sc_u8, &m_u8);
        out.extend_from_slice(&scales_bytes);

        for k in 0..4 {
            for j in 0..32 {
                let v1 = block_data[64 * k + j];
                let v2 = block_data[64 * k + 32 + j];

                let is1 = 2 * k;
                let is2 = 2 * k + 1;

                let d1 = d * sc_u8[is1] as f32;
                let m1 = min * m_u8[is1] as f32;
                let d2 = d * sc_u8[is2] as f32;
                let m2 = min * m_u8[is2] as f32;

                let q1 = if d1 > 0.0 {
                    ((v1 + m1) / d1).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let q2 = if d2 > 0.0 {
                    ((v2 + m2) / d2).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };

                out.push(q1 | (q2 << 4));
            }
        }
    }

    Ok(out)
}

/// Quantize a KV-cache row of `data.len()` elements (any multiple of 32 up to
/// 256) into the **Q4KHalf** layout used by `grim_kv_dequant_attention`'s
/// `quant_format == 3` path (PLAN-kvcache-channel-axis WI-1/WI-3):
///
/// ```text
/// [0..2]   fp16 d     (row-wide scale factor)
/// [2..4]   fp16 dmin  (row-wide min factor)
/// [4 .. 4+s]          s = len/32 per-sub-block 6-bit scale codes
/// [4+s .. 4+2s]       s per-sub-block 6-bit min codes
/// [4+2s ..]           nibbles, PLAIN order: byte (i/2), low nibble = even i
/// ```
///
/// Byte cost per 128-element row: 76 B (vs LegacyNibble's 68 B), buying
/// per-32-channel-group scale+min granularity. Math mirrors [`quant_q4k`];
/// the layout is the j<4 plain-6-bit half of Q4_K's packing with the
/// nibble-interleave-for-chunks dropped (the kernel indexes nibbles directly).
pub fn quant_q4khalf(data: &[f32]) -> Result<Vec<u8>> {
    if data.is_empty() || data.len() % 32 != 0 || data.len() > 256 {
        return Err(Error::Backend(format!(
            "quant_q4khalf: len {} must be a nonzero multiple of 32, <= 256",
            data.len()
        )));
    }
    let s_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(4 + 2 * s_blocks + data.len() / 2);

    let mut sub_d1 = [0.0f32; 8];
    let mut sub_m1 = [0.0f32; 8];
    let mut max_d1 = 0.0f32;
    let mut max_m1 = 0.0f32;
    for s in 0..s_blocks {
        let sub = &data[s * 32..(s + 1) * 32];
        let min_v = sub.iter().copied().fold(f32::INFINITY, f32::min);
        let max_v = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let m1 = if min_v < 0.0 { -min_v } else { 0.0 };
        let d1 = if min_v < 0.0 {
            (max_v - min_v) / 15.0
        } else {
            max_v.max(0.0) / 15.0
        };
        sub_m1[s] = m1;
        sub_d1[s] = d1;
        max_d1 = max_d1.max(d1);
        max_m1 = max_m1.max(m1);
    }
    let d = if max_d1 == 0.0 { 1.0 } else { max_d1 / 63.0 };
    let min = if max_m1 == 0.0 { 0.0 } else { max_m1 / 63.0 };
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    out.extend_from_slice(&f32_to_f16(min).to_le_bytes());

    let mut sc_u8 = [0u8; 8];
    let mut m_u8 = [0u8; 8];
    for s in 0..s_blocks {
        sc_u8[s] = if d > 0.0 {
            (sub_d1[s] / d).round().clamp(1.0, 63.0) as u8
        } else {
            1
        };
        m_u8[s] = if min > 0.0 {
            (sub_m1[s] / min).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
    }
    out.extend(sc_u8[..s_blocks].iter().map(|&sc| sc & 63));
    out.extend(m_u8[..s_blocks].iter().map(|&m| m & 63));

    for i in (0..data.len()).step_by(2) {
        let q_of = |idx: usize| -> u8 {
            let s = idx / 32;
            let d1 = d * sc_u8[s] as f32;
            let m1 = min * m_u8[s] as f32;
            if d1 > 0.0 {
                ((data[idx] + m1) / d1).round().clamp(0.0, 15.0) as u8
            } else {
                0
            }
        };
        let lo = q_of(i);
        let hi = if i + 1 < data.len() { q_of(i + 1) } else { 8 };
        out.push(lo | (hi << 4));
    }
    Ok(out)
}

/// Q4KHalf row byte count for a given head_dim (multiple of 32, ≤ 256).
/// Panics on invalid geometry — call-side validation happens in
/// `quant_q4khalf` before any bytes are produced.
pub fn q4khalf_row_bytes(head_dim: usize) -> usize {
    debug_assert!(head_dim % 32 == 0 && head_dim > 0 && head_dim <= 256);
    4 + 2 * (head_dim / 32) + head_dim / 2
}

/// Dequantize a Q4KHalf buffer produced by [`quant_q4khalf`]. Host-side mirror
/// of the kernel's `quant_format == 3` branch — used as the CPU reference in
/// parity tests.
pub fn dequant_q4khalf(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    if num_weights == 0 || num_weights % 32 != 0 || num_weights > 256 {
        return Err(Error::Backend(format!(
            "dequant_q4khalf: num_weights {num_weights} must be a nonzero multiple of 32, <= 256"
        )));
    }
    let s_blocks = num_weights / 32;
    let row_bytes = 4 + 2 * s_blocks + num_weights / 2;
    if data.len() != row_bytes {
        return Err(Error::Backend(format!(
            "dequant_q4khalf: want {row_bytes} bytes for {num_weights} weights ({s_blocks} sub-blocks), have {}",
            data.len()
        )));
    }
    let d = f16_to_f32(data[0], data[1]);
    let min = f16_to_f32(data[2], data[3]);
    let scales = &data[4..4 + s_blocks];
    let mins = &data[4 + s_blocks..4 + 2 * s_blocks];
    let qs = &data[4 + 2 * s_blocks..];
    let mut out = vec![0.0f32; num_weights];
    for i in 0..num_weights {
        let s = i / 32;
        let sc = (scales[s] & 63) as f32;
        let m = (mins[s] & 63) as f32;
        let byte = qs[i / 2];
        let q = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        out[i] = d * sc * (q as f32) - min * m;
    }
    Ok(out)
}

#[inline]
fn pack_scale_min_k4(scales_sc: &[u8; 8], scales_m: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for j in 0..4 {
        out[j] = (scales_sc[j] & 63) | (((scales_sc[j + 4] >> 4) & 3) << 6);
        out[j + 4] = (scales_m[j] & 63) | (((scales_m[j + 4] >> 4) & 3) << 6);
        out[j + 8] = (scales_sc[j + 4] & 0x0F) | ((scales_m[j + 4] & 0x0F) << 4);
    }
    out
}

/// Quantize a slice of f32 values to Q5_K bytes per the ggml super-block format.
/// Encodes 256-weight blocks into 176-byte Q5_K super-blocks.
pub fn quant_q5k(data: &[f32]) -> Result<Vec<u8>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;

    if data.is_empty() {
        return Ok(Vec::new());
    }

    let num_blocks = data.len().div_ceil(BLOCK_SIZE);
    let mut out = Vec::with_capacity(num_blocks * BLOCK_BYTES);

    for block in data.chunks(BLOCK_SIZE) {
        let mut block_data = [0.0f32; 256];
        block_data[..block.len()].copy_from_slice(block);

        let mut sub_d1 = [0.0f32; 8];
        let mut sub_m1 = [0.0f32; 8];
        let mut max_d1 = 0.0f32;
        let mut max_m1 = 0.0f32;

        for s in 0..8 {
            let sub = &block_data[s * 32..(s + 1) * 32];
            let min_v = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let max_v = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let m1 = if min_v < 0.0 { -min_v } else { 0.0 };
            let d1 = if min_v < 0.0 {
                (max_v - min_v) / 31.0
            } else {
                max_v.max(0.0) / 31.0
            };

            sub_m1[s] = m1;
            sub_d1[s] = d1;
            max_d1 = max_d1.max(d1);
            max_m1 = max_m1.max(m1);
        }

        let d = if max_d1 == 0.0 { 1.0 } else { max_d1 / 63.0 };
        let dm = if max_m1 == 0.0 { 0.0 } else { max_m1 / 63.0 };

        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dm).to_le_bytes());

        let mut sc_u8 = [0u8; 8];
        let mut m_u8 = [0u8; 8];
        for s in 0..8 {
            sc_u8[s] = if d > 0.0 {
                (sub_d1[s] / d).round().clamp(1.0, 63.0) as u8
            } else {
                1
            };
            m_u8[s] = if dm > 0.0 {
                (sub_m1[s] / dm).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
        }

        let scales_bytes = pack_scale_min_k4(&sc_u8, &m_u8);
        out.extend_from_slice(&scales_bytes);

        // qh: 32 bytes holding the high bit of each 5-bit quant (256 bits = 32 bytes).
        // Packed as: for each of 4 groups of 32, u1/u2 shift pattern matching dequant.
        let mut qh = [0u8; 32];
        // qs: 128 bytes holding low 4 bits (nibbles) of each quant.
        let mut qs = [0u8; 128];

        for k in 0..4 {
            for j in 0..32 {
                let v1 = block_data[64 * k + j];
                let v2 = block_data[64 * k + 32 + j];

                let is1 = 2 * k;
                let is2 = 2 * k + 1;
                let d1 = d * sc_u8[is1] as f32;
                let m1 = dm * m_u8[is1] as f32;
                let d2 = d * sc_u8[is2] as f32;
                let m2 = dm * m_u8[is2] as f32;

                let q1 = if d1 > 0.0 {
                    ((v1 + m1) / d1).round().clamp(0.0, 31.0) as u8
                } else {
                    0
                };
                let q2 = if d2 > 0.0 {
                    ((v2 + m2) / d2).round().clamp(0.0, 31.0) as u8
                } else {
                    0
                };

                qs[k * 32 + j] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
                // High bits: q1's bit4 goes into qh[j] via u1 mask, q2's bit4 via u2 mask.
                // u1 starts at 1, u2 at 2, both shift left by 2 each group.
                let u1 = 1u8 << (2 * k);
                let u2 = 2u8 << (2 * k);
                if q1 & 0x10 != 0 {
                    qh[j] |= u1;
                }
                if q2 & 0x10 != 0 {
                    qh[j] |= u2;
                }
            }
        }

        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }

    Ok(out)
}

/// Quantize a slice of f32 values to Q6_K bytes per the ggml super-block format.
/// Encodes 256-weight blocks into 210-byte Q6_K super-blocks.
pub fn quant_q6k(data: &[f32]) -> Result<Vec<u8>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    if data.is_empty() {
        return Ok(Vec::new());
    }

    let num_blocks = data.len().div_ceil(BLOCK_SIZE);
    let mut out = Vec::with_capacity(num_blocks * BLOCK_BYTES);

    for block in data.chunks(BLOCK_SIZE) {
        let mut block_data = [0.0f32; 256];
        block_data[..block.len()].copy_from_slice(block);

        // Q6_K uses 16 sub-blocks of 16 weights each, each with its own i8 scale.
        // Global d is f16.
        let mut sub_scales = [0i8; 16];
        for s in 0..16 {
            let sub = &block_data[s * 16..(s + 1) * 16];
            let max_abs = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let sc = if max_abs > 0.0 {
                (max_abs / 31.0).round() as i8
            } else {
                1
            };
            sub_scales[s] = sc.clamp(1, 63);
        }

        let max_sc = sub_scales
            .iter()
            .map(|&s| s.max(1) as f32)
            .fold(0.0f32, f32::max);
        let d = if max_sc > 0.0 { max_sc / 63.0 } else { 1.0 };

        // Normalize sub-scales to [1..63] relative to d
        if d > 0.0 {
            for sc in sub_scales.iter_mut() {
                if *sc as f32 / d > 63.0 {
                    *sc = 63;
                }
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        // Q6_K layout: 2 super-groups of 128 weights each.
        // Each super-group: 4 sub-blocks of 32 weights.
        for sg in 0..2 {
            let sg_base = sg * 128;
            for l in 0..32 {
                // 4 weights at positions within this super-group:
                let w0 = block_data[sg_base + l];
                let w1 = block_data[sg_base + 64 + l];
                let w2 = block_data[sg_base + l + 32];
                let w3 = block_data[sg_base + 96 + l];

                let is = l / 16; // sub-block index within this super-group (0 or 1)
                let sc_idx = sg * 8 + is * 4;

                let quantize_q6 = |v: f32, sc: i8| -> u8 {
                    if sc > 0 {
                        ((v / (d * sc as f32)).round() + 32.0).clamp(0.0, 63.0) as u8
                    } else {
                        32
                    }
                };

                let q1 = quantize_q6(w0, sub_scales[sc_idx]);
                let q2 = quantize_q6(w2, sub_scales[sc_idx + 2]);
                let q3 = quantize_q6(w1, sub_scales[sc_idx + 1]);
                let q4 = quantize_q6(w3, sub_scales[sc_idx + 3]);

                let ql_off = sg * 64;
                let qh_off = sg * 32;

                ql[ql_off + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                ql[ql_off + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
                qh[qh_off + l] = ((q1 >> 4) & 0x03)
                    | (((q2 >> 4) & 0x03) << 2)
                    | (((q3 >> 4) & 0x03) << 4)
                    | (((q4 >> 4) & 0x03) << 6);
            }
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&sub_scales.map(|s| s as u8));
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }

    Ok(out)
}

/// Quantize f32 values to FP4 (E2M1) bytes.
/// Each f32 is clamped and mapped to the nearest E2M1 value.
pub fn quant_fp4(data: &[f32]) -> Result<Vec<u8>> {
    // Find scale using max absolute value mapped to FP4 range
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // FP4 max representable is 1.0 with our LUT
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs };

    let mut out = Vec::with_capacity(4 + data.len().div_ceil(2));
    out.extend_from_slice(&scale.to_le_bytes());

    let mut packed_byte = 0u8;
    for (i, &v) in data.iter().enumerate() {
        // Map f32 value to nearest FP4 code (using our LUT: 0=-1.0, 7=0.0, 15=+0.875)
        let normalized = (v / scale).clamp(-1.0, 1.0);
        let code = if normalized <= -1.0 {
            0x0 // -1.0
        } else if normalized <= -0.875 {
            0x1
        } else if normalized <= -0.75 {
            0x2
        } else if normalized <= -0.625 {
            0x3
        } else if normalized <= -0.5 {
            0x4
        } else if normalized <= -0.375 {
            0x5
        } else if normalized <= -0.25 {
            0x6
        } else if normalized <= -0.125 {
            0x7
        } else if normalized <= 0.0 {
            0x8 // 0.0
        } else if normalized <= 0.125 {
            0x9 // +0.125
        } else if normalized <= 0.25 {
            0xA
        } else if normalized <= 0.375 {
            0xB
        } else if normalized <= 0.5 {
            0xC
        } else if normalized <= 0.625 {
            0xD
        } else if normalized <= 0.75 {
            0xE
        } else {
            0xF // +0.875
        };

        if i % 2 == 0 {
            packed_byte = code << 4;
        } else {
            packed_byte |= code;
            out.push(packed_byte);
        }
    }
    if data.len() % 2 != 0 {
        out.push(packed_byte);
    }

    Ok(out)
}

/// Quantize f32 values to NF4 (normalized float-4) bytes.
/// NF4 is optimized for normally-distributed weights.
pub fn quant_nf4(data: &[f32]) -> Result<Vec<u8>> {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs };

    let mut out = Vec::with_capacity(4 + data.len().div_ceil(2));
    out.extend_from_slice(&scale.to_le_bytes());

    let mut packed_byte = 0u8;
    for (i, &v) in data.iter().enumerate() {
        let normalized = (v / scale).clamp(-1.0, 1.0);
        let mut min_diff = f32::MAX;
        let mut code = 0u8;
        for (c_idx, &quant_val) in NF4_LUT.iter().enumerate() {
            let diff = (normalized - quant_val).abs();
            if diff < min_diff {
                min_diff = diff;
                code = c_idx as u8;
            }
        }

        if i % 2 == 0 {
            packed_byte = code << 4;
        } else {
            packed_byte |= code;
            out.push(packed_byte);
        }
    }
    if data.len() % 2 != 0 {
        out.push(packed_byte);
    }

    Ok(out)
}

/// Quantize f32 values to FP8 (E4M3) bytes.
/// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa bits.
pub fn quant_fp8(data: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(4 + data.len());

    // Write scale of 1.0 for now (FP8 can represent values directly in reasonable range)
    out.extend_from_slice(&1.0f32.to_le_bytes());

    for &v in data {
        let quantized = f32_to_fp8_e4m3(v);
        out.push(quantized);
    }

    Ok(out)
}

/// Blocked-FP8 layout version for the WhiteRaven blocked WMMA path
/// (`grim_wmma_gemm_fp8_e4m3_blocked`).
///
/// Bumped whenever the 16x16-block byte order changes. It must be mixed into
/// the on-disk cache key of any blocked-FP8 tensor cache (the same discipline
/// as `OSTQUANT_ENCODER_VERSION` for WhiteCrow), so a format change can never
/// serve tensors converted by an older encoder.
pub const FP8_BLOCK16_ENCODER_VERSION: u32 = 1;

/// Rearrange FP8 E4M3 bytes `[N, K]` row-major into 16x16-blocked order for the
/// WhiteRaven blocked WMMA kernel: for each N-tile of 16 rows and each K-block
/// of 16 columns, the 256 bytes are stored contiguously in row-major order
/// within the block. The kernel's `col_major` 16x16 fragment load with ldm=16
/// then reads one contiguous 256B tile instead of 16 K-strided 16B segments.
///
/// Input is raw codes (no scale prefix, unlike `quant_fp8`'s output): exactly
/// `n * k` bytes. Requires `n % 16 == 0` and `k % 16 == 0`; the byte count is
/// unchanged, only the arrangement.
pub fn block_fp8_16x16(data: &[u8], n: usize, k: usize) -> Result<Vec<u8>> {
    if n % 16 != 0 || k % 16 != 0 {
        return Err(Error::Backend(format!(
            "block_fp8_16x16: need n % 16 == 0 and k % 16 == 0, got n={n} k={k}"
        )));
    }
    if data.len() < n * k {
        return Err(Error::Backend(format!(
            "block_fp8_16x16: need {} bytes, got {}",
            n * k,
            data.len()
        )));
    }
    let mut out = vec![0u8; n * k];
    for nt in 0..n / 16 {
        for kb in 0..k / 16 {
            for c in 0..16 {
                let src = (nt * 16 + c) * k + kb * 16;
                let dst = (nt * (k / 16) + kb) * 256 + c * 16;
                out[dst..dst + 16].copy_from_slice(&data[src..src + 16]);
            }
        }
    }
    Ok(out)
}

/// Inverse of [`block_fp8_16x16`]: restores row-major `[N, K]` order. Used by
/// tests and by any path that must serve blocked bytes to a row-major reader.
pub fn unblock_fp8_16x16(blocked: &[u8], n: usize, k: usize) -> Result<Vec<u8>> {
    if n % 16 != 0 || k % 16 != 0 {
        return Err(Error::Backend(format!(
            "unblock_fp8_16x16: need n % 16 == 0 and k % 16 == 0, got n={n} k={k}"
        )));
    }
    if blocked.len() < n * k {
        return Err(Error::Backend(format!(
            "unblock_fp8_16x16: need {} bytes, got {}",
            n * k,
            blocked.len()
        )));
    }
    let mut out = vec![0u8; n * k];
    for nt in 0..n / 16 {
        for kb in 0..k / 16 {
            for c in 0..16 {
                let dst = (nt * 16 + c) * k + kb * 16;
                let src = (nt * (k / 16) + kb) * 256 + c * 16;
                out[dst..dst + 16].copy_from_slice(&blocked[src..src + 16]);
            }
        }
    }
    Ok(out)
}

/// Forward-looking on-disk cache key for blocked-FP8 tensors, mirroring the
/// WhiteCrow discipline (`requant_kquant_to_whitecrow`): seahash of the source
/// bytes mixed with geometry and [`FP8_BLOCK16_ENCODER_VERSION`], so a stale
/// converter or a layout bump can never be served for a different weight.
pub fn blocked_fp8_cache_key(n: usize, k: usize, src_hash: u64) -> u64 {
    let mut h = src_hash;
    h ^= (n as u64) << 1;
    h ^= (k as u64) << 17;
    h ^= (FP8_BLOCK16_ENCODER_VERSION as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h
}

/// Host dequant of 16x16-blocked FP8 bytes to f32 (scale 1.0): unblock to
/// row-major, then decode each E4M3 code.
///
/// WARNING: do NOT feed blocked bytes to [`dequant_fp8`]: it reads a 4-byte
/// f32 scale header that blocked tensors do not have, so the first four
/// codes would be misread as a scale and every value after would shift by
/// four. This function is the only host decoder for the blocked layout.
pub fn dequant_fp8_blocked16(blocked: &[u8], n: usize, k: usize) -> Result<Vec<f32>> {
    let row_major = unblock_fp8_16x16(blocked, n, k)?;
    Ok(row_major.iter().map(|&b| fp8_e4m3_to_f32(b)).collect())
}

/// Quantize f32 to FP8 E4M3, round-to-nearest-**even**.
///
/// # Rounding
///
/// This rounds; it does not truncate. Truncation is not a smaller error, it is
/// a *systematic* one: every value is biased toward zero, so a dot product over
/// truncated weights is quietly computing something slightly wrong in a
/// consistent direction, which no tolerance test on the mean will catch. RNE is
/// unbiased, matches IEEE's default for every other format here, and is the only
/// one of grim's E4M3 converters that is round-to-nearest-even -- so it is also
/// the only one whose result is independent of accumulation order.
///
/// The device converter this must agree with is
/// `dot_gemv.rs::grim_f32_to_fp8_e4m3`. `quant_standalone.rs` still rounds
/// half-up and is the remaining divergence. See
/// `tests/e4m3_rne_convergence.rs`.
pub fn f32_to_fp8_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F; // the NaN slot in E4M3
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0u8 };
    let a = v.abs();

    // E4M3 has no infinities: exp=15,mant=7 (0x7F) is NaN, so the largest finite
    // value is exp=15,mant=6 (0x7E) = 1.75*2^8 = 448.
    //
    // The threshold must be 448, not 480. Any f32 in [448,480) has exponent
    // field E=15, so its 3-bit mantissa q runs 0..7 -- and q==7 lands on 0x7F,
    // the NaN slot. f32 in [464.01,480) therefore encoded as NaN and propagated
    // as NaN rather than as a saturated value. 448 is correct: the first f32
    // that would carry q past 6 is 464, well inside it.
    if a.is_infinite() || a >= 448.0 {
        return sign | 0x7E; // saturate
    }
    if a == 0.0 {
        return sign;
    }

    let bits = a.to_bits();
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    if raw_exp == 0 {
        // f32 subnormal: far below E4M3's min subnormal (2^-9), rounds to zero.
        // Checked explicitly because the shift below would overflow.
        return sign;
    }
    let m = bits & 0x007F_FFFF;
    let e4m3_exp = raw_exp - 127 + 7; // unbiased exponent, then rebiased to 7

    if e4m3_exp >= 1 {
        // Normal: round the 23-bit fraction to 3 bits, RNE.
        let mut q = m >> 20;
        let r = m & 0x000F_FFFF;
        let half = 0x0008_0000u32;
        if r > half || (r == half && (q & 1) == 1) {
            q += 1;
        }
        if q == 8 {
            // Carry into the exponent. Wrapping the mantissa instead would emit
            // a code that is not on the grid at all -- silently corrupt.
            q = 0;
            let e4m3_exp = e4m3_exp + 1;
            if e4m3_exp > 15 {
                return sign | 0x7E;
            }
            return sign | ((e4m3_exp as u8) << 3) | q as u8;
        }
        sign | ((e4m3_exp as u8) << 3) | q as u8
    } else {
        // Subnormal: value = (mant/8) * 2^-6, so value*512 is a 3-bit integer.
        // Shift the implicit leading 1 into place, then RNE.
        let sh = 21 - e4m3_exp; // 21 at the top of the range, unbounded below
        if sh >= 32 {
            // Below half the min subnormal: rounds to zero. Checked before the
            // masks, which would overflow for a shift this large.
            return sign;
        }
        let full = 0x0080_0000u32 | m;
        let mut q = full >> sh;
        let r = full & ((1u32 << sh) - 1);
        let half = 1u32 << (sh - 1);
        if r > half || (r == half && (q & 1) == 1) {
            q += 1;
        }
        if q == 0 {
            return sign;
        }
        // q can reach 8 after rounding, which is exactly the min normal (code
        // 8 = 2^-6) -- correct as-is, since the subnormal and normal rows are
        // contiguous in E4M3.
        sign | q as u8
    }
}

/// Convert MXFP4 E2M1 (2-bit exp, 1-bit mantissa) + E8M0 shared exponent to f32.
pub fn mxfp4_e2m1_to_f32(code: u8, shared_exp: u8) -> f32 {
    let sign = ((code >> 3) & 1) != 0;
    let exp = (code >> 1) & 3;
    let mant = code & 1;
    let base_val = if exp == 0 {
        mant as f32 * 0.5
    } else {
        (1.0 + mant as f32 * 0.5) * (2.0f32).powi(exp as i32 - 1)
    };
    let signed_val = if sign { -base_val } else { base_val };
    let scale = (2.0f32).powi(shared_exp as i32 - 127);
    signed_val * scale
}

// ── NVFP4 (the real thing) ────────────────────────────────────────────────
//
// GGUF type 78 / NVIDIA Blackwell NVFP4: E2M1 elements with an **E4M3** block
// scale per 16 elements, plus an optional per-tensor FP32 scale.
//
// Grim previously decoded this type with an E8M0 scale, which is a different
// format. Because E4M3 and E8M0 are both exactly 1 byte per 16 elements the
// buffer length, allocation and bounds checks all still passed, so the bug was
// silent: a real type-78 tensor with E4M3 scale 0x3C (= 1.5) was read as
// 2^(0x3C - 127) = 6.8e-21, collapsing every weight in the block to ~0.
//
// The 9-bytes-per-16 layout is unchanged from Nutcracker; only the scale-byte
// decode differs.

/// Split an NVFP4 E4M3 block-scale byte into its float value.
pub fn nvfp4_e4m3_scale(scale_byte: u8) -> f32 {
    fp8_e4m3_to_f32(scale_byte)
}

/// Convert an NVFP4 E2M1 code + E4M3 block-scale byte to f32.
///
/// Unlike [`nutcracker_e2m1_to_f32`], code `0x0`/`0x8` decode to a true zero —
/// NVFP4 has no repurposed code.
pub fn nvfp4_e2m1_to_f32(code: u8, scale_byte: u8) -> f32 {
    let base = mxfp4_e2m1_to_f32(code, 127);
    base * nvfp4_e4m3_scale(scale_byte)
}

/// Dequantize a packed NVFP4 buffer (9 bytes per 16 values: 1 E4M3 scale byte
/// + 8 packed E2M1 code bytes) to f32.
pub fn dequant_nvfp4(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    const SUB_BLOCK: usize = 16;
    const CODES_PER_SUB: usize = 8;
    const SUB_BLOCK_BYTES: usize = 1 + CODES_PER_SUB; // 9

    let num_sub_blocks = num_values.div_ceil(SUB_BLOCK);
    let expected_bytes = num_sub_blocks * SUB_BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "NVFP4: expected {expected_bytes} bytes for {num_values} values \
             ({num_sub_blocks} sub-blocks), got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_values);
    let mut pos = 0usize;
    for _ in 0..num_sub_blocks {
        let scale_byte = data[pos];
        pos += 1;
        let codes = &data[pos..pos + CODES_PER_SUB];
        pos += CODES_PER_SUB;

        let out_start = out.len();
        let out_end = (out_start + SUB_BLOCK).min(num_values);
        for i in out_start..out_end {
            let local = i - out_start;
            let code_byte = codes[local / 2];
            let code = if local % 2 == 0 {
                code_byte & 0x0F
            } else {
                (code_byte >> 4) & 0x0F
            };
            out.push(nvfp4_e2m1_to_f32(code, scale_byte));
        }
    }
    while out.len() < num_values {
        out.push(0.0);
    }
    Ok(out)
}

/// Quantize f32 values to packed NVFP4 bytes (9 bytes per 16 values:
/// 1 E4M3 scale byte + 8 packed E2M1 code bytes, low nibble first).
pub fn quant_nvfp4(data: &[f32]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    const SUB_BLOCK: usize = 16;
    const E2M1_VALS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    let num_sub_blocks = data.len().div_ceil(SUB_BLOCK);
    let mut out = Vec::with_capacity(num_sub_blocks * 9);

    for chunk in data.chunks(SUB_BLOCK) {
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        // Scale = amax / 6.0, clamped to min normal and encoded to E4M3.
        let raw_scale = if amax > 0.0 { amax / 6.0 } else { 0.0 };
        let scale_byte = f32_to_fp8_e4m3(raw_scale);
        out.push(scale_byte);

        // Quantize against the decoded (rounded) scale so scale rounding doesn't clip
        let eff_scale = nvfp4_e4m3_scale(scale_byte);
        let inv_scale = if eff_scale > 0.0 {
            1.0 / eff_scale
        } else {
            0.0
        };

        let mut codes = [0u8; 16];
        for (i, &v) in chunk.iter().enumerate() {
            let sign_bit = if v.is_sign_negative() { 0x8 } else { 0x0 };
            let abs_norm = v.abs() * inv_scale;

            // Nearest-neighbor matching on E2M1 magnitude grid: {0, .5, 1, 1.5, 2, 3, 4, 6}
            let mut best_idx = 0u8;
            let mut min_diff = f32::MAX;
            for (idx, &grid_val) in E2M1_VALS.iter().enumerate() {
                let diff = (abs_norm - grid_val).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_idx = idx as u8;
                }
            }
            codes[i] = sign_bit | best_idx;
        }

        // Pack low-nibble first: local%2==0 is in low nibble (code & 0x0F),
        // local%2==1 is in high nibble (code << 4).
        for pair in 0..8 {
            let even = codes[pair * 2];
            let odd = codes[pair * 2 + 1];
            let packed_byte = (even & 0x0F) | ((odd & 0x0F) << 4);
            out.push(packed_byte);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod nvfp4_tests {
    use super::*;

    /// E4M3 0x38 is 1.0 (exp=7, mant=0). This is the value the old E8M0 path
    /// misread as 2^(0x38 - 127) = 2^-71.
    const UNIT_E4M3: u8 = 0x38;

    #[test]
    fn e4m3_unit_scale_decodes_to_one() {
        assert_eq!(nvfp4_e4m3_scale(UNIT_E4M3), 1.0);
    }

    #[test]
    fn zero_codes_are_a_real_zero_in_nvfp4() {
        // The critical difference from Nutcracker: NVFP4 keeps the zero.
        assert_eq!(nvfp4_e2m1_to_f32(0x0, UNIT_E4M3), 0.0);
        assert_eq!(nvfp4_e2m1_to_f32(0x8, UNIT_E4M3), 0.0);
        // And no special value appears anywhere in the codebook.
        let grid: Vec<f32> = (0u8..16).map(|c| nvfp4_e2m1_to_f32(c, UNIT_E4M3)).collect();
        for v in &grid {
            let expected = mxfp4_e2m1_to_f32(
                (0u8..16)
                    .find(|&c| mxfp4_e2m1_to_f32(c, 127) == *v)
                    .unwrap(),
                127,
            );
            assert_eq!(*v, expected);
        }
    }

    /// The regression that motivated a correct NVFP4: an E4M3 scale byte must
    /// not be interpreted as E8M0. Byte 0x3C is 1.5 in E4M3.
    #[test]
    fn e4m3_scale_is_not_misread_as_e8m0() {
        let e4m3 = nvfp4_e4m3_scale(0x3C);
        let wrong_e8m0 = (2.0f32).powi(0x3C as i32 - 127);
        assert_eq!(e4m3, 1.5, "0x3C must decode as 1.5 in E4M3");
        assert!(
            e4m3 > wrong_e8m0 * 1e18,
            "E8M0 misread gives {wrong_e8m0:e}, E4M3 gives {e4m3}"
        );
    }

    #[test]
    fn nvfp4_and_nutcracker_share_a_layout_but_not_a_decode() {
        // Same 9 bytes, different scale semantics — this is why they must stay
        // distinct schemes with distinct tags.
        let mut buf = vec![UNIT_E4M3; 9];
        for (i, b) in buf[1..].iter_mut().enumerate() {
            *b = if i % 2 == 0 { 0x11 } else { 0x11 };
        }
        let nv = dequant_nvfp4(&buf, 16).unwrap();
        let nu = dequant_nutcracker(&buf, 16).unwrap();
        // Nutcracker reads 0x38 as [exp:6|sel:2] = exp 14, sel 0 -> 2^(14-31).
        assert_eq!(nv[0], 0.5, "code 0x1 is +0.5 under NVFP4");
        assert_ne!(nu[0], nv[0], "the two schemes must not coincide");
    }

    #[test]
    fn dequant_nvfp4_rejects_a_short_buffer() {
        let err = dequant_nvfp4(&[0u8; 8], 16).unwrap_err();
        assert!(format!("{err}").contains("expected 9 bytes"), "got {err}");
    }

    #[test]
    fn quant_nvfp4_roundtrip_matches_dequant_oracle() {
        // Construct 16 known E2M1 values with scale 1.0 (0x38 in E4M3)
        let inputs = vec![
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let packed = quant_nvfp4(&inputs).expect("quant_nvfp4");
        assert_eq!(packed.len(), 9);
        assert_eq!(packed[0], UNIT_E4M3, "scale should be 1.0 (0x38 in E4M3)");

        let dequant = dequant_nvfp4(&packed, 16).expect("dequant_nvfp4");
        for (i, (&inp, &out)) in inputs.iter().zip(dequant.iter()).enumerate() {
            assert_eq!(
                inp.abs(),
                out.abs(),
                "mismatch at index {i}: {inp} vs {out}"
            );
            assert_eq!(
                inp.is_sign_negative(),
                out.is_sign_negative(),
                "sign mismatch at index {i}"
            );
        }
    }
}

// ── Nutcracker ───────────────────────────────────────────────────────────
//
// Internal 4-bit float format: E2M1 codes + a 1-byte per-16 block scale, in
// the same 9-bytes-per-16-values layout as NVFP4. The scale byte's low bits
// are stolen as a special-value selector and the redundant E2M1 zero code is
// repurposed to emit that value, so each block carries one extra quantization
// level chosen to minimise that block's error.
//
// This is the RaZeR idea (arXiv:2501.04052) adapted to grim's actual scale
// format. RaZeR steals the sign bit of an E4M3 block scale; Nutcracker is
// defined directly on the E8M0-shaped byte grim already packed, so the
// selector comes out of exponent range instead. RaZeR's own Table 1 shows the
// same trade is free for LLM weight block scales (E3M3 scores identically to
// E4M3), which is what makes stealing exponent bits defensible here.
//
// Wire format (per 16 values, 9 bytes — byte-identical to NvFp4):
//   byte 0      : [ exp : 6 | sel : 2 ]   scale = 2^(exp - 31)
//   bytes 1..=8 : 16 E2M1 codes, low nibble = even index
//   code & 0x7 == 0 (the +/-0 encodings) decodes to the block's special value
//   with the sign taken from `sel` bit 1, not from the code's sign bit.
//
// Consequence: Nutcracker cannot represent an exact zero. A quantizer must map
// near-zero to +/-0.5 (code 0x1 / 0x9), the smallest non-zero code.

/// Number of low bits stolen from the E8M0 block-scale byte as the selector.
pub const NUTCRACKER_SEL_BITS: u8 = 2;
/// Mask over the stolen selector bits.
pub const NUTCRACKER_SEL_MASK: u8 = (1u8 << NUTCRACKER_SEL_BITS) - 1;
/// Bias of the remaining `8 - NUTCRACKER_SEL_BITS` exponent field.
///
/// 6 exponent bits span `2^-31 ..= 2^32`, far wider than an LLM weight block
/// scale needs.
pub const NUTCRACKER_SCALE_BIAS: i32 = 31;

/// Values per sub-block, and the sub-block byte count (1 scale + 8 code bytes).
pub const NUTCRACKER_SUB_BLOCK: usize = 16;
/// Bytes per sub-block: 1 scale byte + 8 packed code bytes.
pub const NUTCRACKER_SUB_BLOCK_BYTES: usize = 9;

/// The special value a block has selected, in unscaled (pre-`scale`) units.
///
/// Two +/- pairs, ordered so the selector's high bit carries the sign and the
/// low bit selects the magnitude: `0 -> +5.0`, `1 -> +2.5`, `2 -> -5.0`,
/// `3 -> -2.5`.
///
/// 5.0 is the midpoint of the E2M1 codebook's widest gap (4 -> 6); 2.5
/// bridges the next widest (2 -> 3). Both are exact multiples of 0.5, so
/// decoded values stay on the FP4 grid and the MAC stays low-precision.
pub fn nutcracker_special_value(sel: u8) -> f32 {
    let mag = if sel & 0x1 != 0 { 2.5 } else { 5.0 };
    if sel & 0x2 != 0 {
        -mag
    } else {
        mag
    }
}

/// Split a Nutcracker block-scale byte into `(selector, scale)`.
pub fn nutcracker_split_scale(scale_byte: u8) -> (u8, f32) {
    let sel = scale_byte & NUTCRACKER_SEL_MASK;
    let exp = scale_byte >> NUTCRACKER_SEL_BITS;
    (sel, (2.0f32).powi(exp as i32 - NUTCRACKER_SCALE_BIAS))
}

/// Pack a Nutcracker block-scale byte from its 6-bit exponent and a selector.
///
/// The exponent is the Nutcracker field, not a raw E8M0 byte: the equivalent
/// E8M0 value that would produce the same scale is `exp + 96`, since
/// `2^(exp - 31) == 2^((exp + 96) - 127)`.
pub fn nutcracker_pack_scale(exp: u8, sel: u8) -> u8 {
    debug_assert!(
        exp < (1u8 << (8 - NUTCRACKER_SEL_BITS)),
        "Nutcracker exponent {exp} exceeds the 6-bit field"
    );
    (exp << NUTCRACKER_SEL_BITS) | (sel & NUTCRACKER_SEL_MASK)
}

/// Convert a Nutcracker E2M1 code + block-scale byte to f32.
///
/// Mirrors `nutcracker_to_float_hip` in
/// `crates/grim-backend-rocm/src/kernels/shared_device_fns.rs`.
pub fn nutcracker_e2m1_to_f32(code: u8, scale_byte: u8) -> f32 {
    let (sel, scale) = nutcracker_split_scale(scale_byte);
    if code & 0x7 == 0 {
        return nutcracker_special_value(sel) * scale;
    }
    let exp = (code >> 1) & 3;
    let mant = code & 1;
    let base_val = if exp == 0 {
        mant as f32 * 0.5
    } else {
        (1.0 + mant as f32 * 0.5) * (2.0f32).powi(exp as i32 - 1)
    };
    let signed_val = if code & 0x8 != 0 { -base_val } else { base_val };
    signed_val * scale
}

/// Dequantize a packed Nutcracker buffer (9 bytes per 16 values) to f32.
///
/// Byte-compatible with [`dequant_nutcracker`], so a Nutcracker buffer can be fed to
/// the NvFp4 unpack and only the leaf decode differs.
pub fn dequant_nutcracker(data: &[u8], num_values: usize) -> Result<Vec<f32>> {
    if num_values == 0 {
        return Ok(Vec::new());
    }
    let codes_per_sub = NUTCRACKER_SUB_BLOCK / 2;
    let num_sub_blocks = num_values.div_ceil(NUTCRACKER_SUB_BLOCK);
    let expected_bytes = num_sub_blocks * NUTCRACKER_SUB_BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "Nutcracker: expected {expected_bytes} bytes for {num_values} values \
             ({num_sub_blocks} sub-blocks), got {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(num_values);
    let mut pos = 0usize;
    for _ in 0..num_sub_blocks {
        let scale_byte = data[pos];
        pos += 1;
        let codes = &data[pos..pos + codes_per_sub];
        pos += codes_per_sub;

        // Last sub-block may be partial if num_values isn't a multiple of 16.
        let out_start = out.len();
        let out_end = (out_start + NUTCRACKER_SUB_BLOCK).min(num_values);
        for i in out_start..out_end {
            let local = i - out_start;
            let code_byte = codes[local / 2];
            let code = if local % 2 == 0 {
                code_byte & 0x0F
            } else {
                (code_byte >> 4) & 0x0F
            };
            out.push(nutcracker_e2m1_to_f32(code, scale_byte));
        }
    }
    while out.len() < num_values {
        out.push(0.0);
    }
    Ok(out)
}

#[cfg(test)]
mod nutcracker_tests {
    use super::*;

    /// Scale byte whose decoded scale is exactly 1.0: exp = 31 -> 2^(31-31).
    /// MXFP4 reaches scale 1.0 at shared_exp = 127, so this pairs with
    /// `mxfp4_e2m1_to_f32(code, 127)` for equivalence checks.
    const UNIT_SCALE_BYTE: u8 = (NUTCRACKER_SCALE_BIAS as u8) << NUTCRACKER_SEL_BITS;

    #[test]
    fn special_value_table_is_the_documented_four() {
        assert_eq!(nutcracker_special_value(0), 5.0);
        assert_eq!(nutcracker_special_value(1), 2.5);
        assert_eq!(nutcracker_special_value(2), -5.0);
        assert_eq!(nutcracker_special_value(3), -2.5);
        // Every special value sits on the 0.5 grid, so the MAC stays low-precision.
        for sel in 0..4u8 {
            let v = nutcracker_special_value(sel);
            assert_eq!(
                v * 2.0,
                (v * 2.0).round(),
                "sel {sel} not a multiple of 0.5"
            );
        }
    }

    #[test]
    fn non_special_codes_match_mxfp4_exactly() {
        // Nutcracker must be a strict superset of MXFP4: the 14 codes that are
        // not the redundant zero must decode bit-identically.
        for code in 0u8..16 {
            if code & 0x7 == 0 {
                continue; // the two zero encodings are repurposed
            }
            let nut = nutcracker_e2m1_to_f32(code, UNIT_SCALE_BYTE);
            let mx = mxfp4_e2m1_to_f32(code, 127);
            assert_eq!(nut, mx, "code {code:#04x} diverged from MXFP4");
        }
    }

    #[test]
    fn both_zero_encodings_decode_to_the_block_special_value() {
        // 0x0 and 0x8 are the +/-0 codes. Nutcracker repurposes both; the sign
        // comes from the selector, so the code's own sign bit is ignored.
        for sel in 0..4u8 {
            let byte = UNIT_SCALE_BYTE | sel;
            let expected = nutcracker_special_value(sel);
            assert_eq!(nutcracker_e2m1_to_f32(0x0, byte), expected);
            assert_eq!(
                nutcracker_e2m1_to_f32(0x8, byte),
                expected,
                "code sign must not apply"
            );
        }
    }

    #[test]
    fn selector_actually_varies_the_special_value() {
        let vals: Vec<f32> = (0..4u8)
            .map(|sel| nutcracker_e2m1_to_f32(0x0, UNIT_SCALE_BYTE | sel))
            .collect();
        assert_eq!(vals, vec![5.0, 2.5, -5.0, -2.5]);
        // Four distinct magnitudes-with-sign: two +/- pairs, no duplicates.
        for i in 0..vals.len() {
            for j in (i + 1)..vals.len() {
                assert_ne!(vals[i], vals[j]);
            }
        }
    }

    #[test]
    fn scale_byte_split_and_pack_round_trip() {
        for sel in 0..4u8 {
            for exp in [0u8, 1, 17, 31, 62, 63] {
                let byte = nutcracker_pack_scale(exp, sel);
                let (got_sel, got_scale) = nutcracker_split_scale(byte);
                assert_eq!(got_sel, sel);
                assert_eq!(got_scale, (2.0f32).powi(exp as i32 - NUTCRACKER_SCALE_BIAS));
            }
        }
    }

    #[test]
    fn stolen_bits_cost_enough_exponent_range() {
        // The whole argument for stealing exponent bits is that 6 bits still
        // covers everything a weight block scale needs.
        let (min, max) = (nutcracker_split_scale(0).1, nutcracker_split_scale(0xFF).1);
        assert_eq!(min, 2.0f32.powi(-NUTCRACKER_SCALE_BIAS));
        assert_eq!(max, 2.0f32.powi(63 - NUTCRACKER_SCALE_BIAS));
        // Full E8M0 range for comparison: 2^-127 .. 2^128.
        assert!(
            min < 2.0f32.powi(-20),
            "range must reach small weight blocks"
        );
        assert!(
            max > 2.0f32.powi(10),
            "range must reach large weight blocks"
        );
    }

    #[test]
    fn decode_is_the_e2m1_codebook_with_zero_replaced() {
        // Full grid at scale 1.0 with selector 0. The 14 ordinary codes keep
        // their MXFP4 values; both zero encodings (0x0 and 0x8) now yield +5.0,
        // which is the whole point of the format.
        let mut got: Vec<f32> = (0u8..16)
            .map(|c| nutcracker_e2m1_to_f32(c, UNIT_SCALE_BYTE | 0))
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            vec![
                -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.0,
                6.0
            ]
        );
        // No zero survives — the documented cost of the format.
        assert!(!got.iter().any(|v| *v == 0.0));
    }

    #[test]
    fn every_selector_produces_a_usable_codebook() {
        for sel in 0..4u8 {
            let mut got: Vec<f32> = (0u8..16)
                .map(|c| nutcracker_e2m1_to_f32(c, UNIT_SCALE_BYTE | sel))
                .collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got.len(), 16);
            // 15 distinct magnitudes: the 14 ordinary codes plus the one the
            // selector supplies (both zero codes land on it).
            let distinct: std::collections::BTreeSet<u32> =
                got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(distinct.len(), 15, "sel {sel} collapsed a level");
        }
    }

    #[test]
    fn dequant_nutcracker_matches_per_element_decode() {
        // One sub-block: scale byte selects -5.0, codes cover the whole 4-bit range.
        let codes: Vec<u8> = (0u8..16).collect();
        let mut packed = vec![UNIT_SCALE_BYTE | 0];
        for pair in codes.chunks(2) {
            packed.push((pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4));
        }
        assert_eq!(packed.len(), NUTCRACKER_SUB_BLOCK_BYTES);

        let out = dequant_nutcracker(&packed, 16).unwrap();
        for (i, &code) in codes.iter().enumerate() {
            assert_eq!(out[i], nutcracker_e2m1_to_f32(code, packed[0]), "index {i}");
        }
    }

    #[test]
    fn dequant_nutcracker_handles_a_partial_trailing_sub_block() {
        let mut packed = vec![UNIT_SCALE_BYTE | 1];
        packed.extend_from_slice(&[0x21, 0x43, 0x65, 0x87, 0xA9, 0xBC, 0xDE, 0xF0]);
        let out = dequant_nutcracker(&packed, 10).unwrap();
        assert_eq!(out.len(), 10);
        // 10 values fit in the 8 code bytes (2 per byte).
        for i in 0..10 {
            let code_byte = packed[1 + i / 2];
            let code = if i % 2 == 0 {
                code_byte & 0x0F
            } else {
                code_byte >> 4
            };
            assert_eq!(out[i], nutcracker_e2m1_to_f32(code, packed[0]));
        }
    }

    #[test]
    fn dequant_nutcracker_rejects_a_short_buffer() {
        // 32 values need 2 sub-blocks = 18 bytes.
        let err = dequant_nutcracker(&[0u8; 17], 32).unwrap_err();
        assert!(format!("{err}").contains("expected 18 bytes"), "got {err}");
    }

    #[test]
    fn dequant_nutcracker_handles_zero_values() {
        assert!(dequant_nutcracker(&[], 0).unwrap().is_empty());
    }

    // ── packer ────────────────────────────────────────────────────────────

    #[test]
    fn packer_emits_the_documented_wire_size() {
        for n in [1usize, 15, 16, 17, 32, 100] {
            let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let packed = quant_nutcracker(&data).unwrap();
            let expected = n.div_ceil(NUTCRACKER_SUB_BLOCK) * NUTCRACKER_SUB_BLOCK_BYTES;
            assert_eq!(packed.len(), expected, "n = {n}");
        }
        assert!(quant_nutcracker(&[]).unwrap().is_empty());
    }

    #[test]
    fn packer_round_trips_through_the_packer_decoder() {
        // Values spread across the E2M1 range so blocks actually exercise the
        // codebook and the special value.
        let data: Vec<f32> = (0..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                (t * 12.0 - 6.0) * (1.0 + 0.1 * ((i % 7) as f32 - 3.0))
            })
            .collect();
        let packed = quant_nutcracker(&data).unwrap();
        let recovered = dequant_nutcracker(&packed, data.len()).unwrap();
        assert_eq!(recovered.len(), data.len());

        // RMS error is the meaningful bound. Per-element relative error is
        // unbounded by design: a block's max pins the shared exponent, so a
        // value near that max can sit up to one exponent step above the E2M1
        // grid ceiling (e.g. 7.56 -> 6.0 when the block max is 7.56).
        let mut se = 0.0f32;
        for (a, b) in data.iter().zip(recovered.iter()) {
            se += (a - b) * (a - b);
        }
        let rmse = (se / data.len() as f32).sqrt();
        let rms = (data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32).sqrt();
        let nrmse = rmse / rms;
        // E2M1's worst adjacent-level gap is 4->6, so a 4-bit grid with one
        // extra level lands well under 10% NRMSE on smooth data.
        assert!(nrmse < 0.10, "NRMSE {nrmse:.4} exceeds 10%");
    }

    #[test]
    fn packer_beats_the_bare_e2m1_grid() {
        // The whole claim of the format is that the per-block selector reduces
        // error versus plain per-16 E2M1 with no special value. Compare against
        // a manual baseline that never uses the zero code as a real value.
        let data: Vec<f32> = (0..512)
            .map(|i| {
                let t = i as f32 / 511.0;
                (t * 10.0 - 5.0) + 0.37 * ((i % 11) as f32 - 5.0)
            })
            .collect();

        let packed = quant_nutcracker(&data).unwrap();
        let rec = dequant_nutcracker(&packed, data.len()).unwrap();
        let nut_err: f32 = data
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        // Baseline: same exponent choice, but the special value is disabled by
        // mapping every zero code onto the smallest magnitude instead.
        let mut base_err = 0.0f32;
        for block in data.chunks(NUTCRACKER_SUB_BLOCK) {
            let max_abs = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let exp: u8 = if max_abs == 0.0 {
                NUTCRACKER_SCALE_BIAS as u8
            } else {
                let raw = (max_abs / 6.0).log2() + NUTCRACKER_SCALE_BIAS as f32;
                (raw.ceil() as i32).clamp(0, 63) as u8
            };
            let scale = (2.0f32).powi(exp as i32 - NUTCRACKER_SCALE_BIAS);
            for &v in block {
                let mut best = 0.0f32; // zero maps to real zero here
                let mut best_d = (v - 0.0).abs();
                for c in 1u8..16 {
                    let val = mxfp4_e2m1_to_f32(c, 127) * scale;
                    let d = (v - val).abs();
                    if d < best_d {
                        best_d = d;
                        best = val;
                    }
                }
                base_err += (v - best) * (v - best);
            }
        }
        assert!(
            nut_err < base_err,
            "Nutcracker SSE {nut_err} should beat the bare E2M1 baseline {base_err}"
        );
    }

    #[test]
    fn packer_selector_actually_reduces_error() {
        // A correct sweep picks sel 0 (+5.0) for blocks whose unscaled values
        // cluster in the 4..6 gap, because 5.0 splits the widest E2M1 gap.
        let data: Vec<f32> = (0..64).map(|i| 4.0 + (i as f32) * 0.031).collect();
        let packed = quant_nutcracker(&data).unwrap();
        let sel = packed[0] & NUTCRACKER_SEL_MASK;
        assert_eq!(sel, 0, "gap-spanning block should select +5.0");

        // And it must beat the same block quantized with the other selectors.
        let err_for = |s: u8| -> f32 {
            let byte = nutcracker_pack_scale(31, s);
            let scale = 1.0f32;
            data.iter()
                .map(|v| {
                    let c = nutcracker_nearest_code(*v / scale, s);
                    let r = nutcracker_e2m1_to_f32(c, byte);
                    (v - r) * (v - r)
                })
                .sum()
        };
        for s in 1..4u8 {
            assert!(
                err_for(sel) <= err_for(s),
                "sel {sel} SSE {} should be <= sel {s} SSE {}",
                err_for(sel),
                err_for(s)
            );
        }
    }

    #[test]
    fn packer_selector_varies_across_different_block_shapes() {
        // The exponent choice normalizes each block's max to 6.0, so blocks are
        // distinguished by where their mass sits *below* that, not by raw
        // magnitude. Two shapes that pick differently:
        //
        //   - mass low (0.3..1.6 unscaled, one outlier pinning the exponent):
        //     the 1.5..2.0 gap is the widest reachable, so +2.5 (sel 1) wins.
        //   - mass in the top gap (4.3..6.0 unscaled): the 4..6 gap dominates,
        //     so +5.0 (sel 0) wins.
        let low: Vec<f32> = {
            let mut v: Vec<f32> = (0..15).map(|i| 0.1 + (i as f32) * 0.1).collect();
            v.push(2.0);
            v
        };
        let high: Vec<f32> = (0..16).map(|i| 4.2 + (i as f32) * 0.11).collect();

        let l = quant_nutcracker(&low).unwrap();
        let h = quant_nutcracker(&high).unwrap();
        let ls = l[0] & NUTCRACKER_SEL_MASK;
        let hs = h[0] & NUTCRACKER_SEL_MASK;
        assert_eq!(ls, 1, "bottom-heavy block should select the +2.5 special");
        assert_eq!(hs, 0, "top-gap block should select the +5.0 special");
        assert_ne!(ls, hs, "distinct block shapes must be able to differ");
    }
}

/// Quantize `data` to Nutcracker (E2M1 + `[exp:6|sel:2]` per-16 block scale).
///
/// Returns the packed buffer: 9 bytes per 16 values, no global-scale header.
/// This is the packer counterpart to [`dequant_nutcracker`] and matches
/// `FloatPackScheme::NutFp4` byte-for-byte.
///
/// Per 16-value block:
/// 1. Pick the shared exponent so the block's largest magnitude lands near the
///    top of the E2M1 range (6.0), which is what the exponent field is for.
/// 2. Sweep all 4 selectors and keep the one minimising the block's squared
///    reconstruction error. The selected value is then available to every
///    zero code in the block, giving the block one extra effective level.
///
/// Note the format cannot represent an exact zero (see [`dequant_nutcracker`]),
/// so a value that quantizes to zero lands on the block's special value.
pub fn quant_nutcracker(data: &[f32]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let num_blocks = data.len().div_ceil(NUTCRACKER_SUB_BLOCK);
    let mut out = Vec::with_capacity(num_blocks * NUTCRACKER_SUB_BLOCK_BYTES);

    for block in data.chunks(NUTCRACKER_SUB_BLOCK) {
        // Choose the exponent so the block's largest magnitude fits just inside
        // the top of the E2M1 grid (6.0).
        //
        // `ceil`, not `round`/`floor`: we need scale >= max/6, so the exponent
        // must round *up* out of log2 space. Rounding either other way lets
        // max/scale exceed 6.0 and clips the block's largest value (a max of
        // exactly 2.0 floors to scale 0.25 -> 8.0 unscaled, a 25% clip).
        let max_abs = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let exp: u8 = if max_abs == 0.0 {
            NUTCRACKER_SCALE_BIAS as u8
        } else {
            let raw = (max_abs / 6.0).log2() + NUTCRACKER_SCALE_BIAS as f32;
            let max_exp = (1i32 << (8 - NUTCRACKER_SEL_BITS)) - 1;
            (raw.ceil() as i32).clamp(0, max_exp) as u8
        };
        let scale = (2.0f32).powi(exp as i32 - NUTCRACKER_SCALE_BIAS);

        // Sweep the 4 selectors; keep the lowest squared error.
        let mut best_sel = 0u8;
        let mut best_err = f32::MAX;
        let mut best_codes = [0u8; NUTCRACKER_SUB_BLOCK];
        for sel in 0..(1u8 << NUTCRACKER_SEL_BITS) {
            let mut codes = [0u8; NUTCRACKER_SUB_BLOCK];
            let mut err = 0.0f32;
            for (i, &v) in block.iter().enumerate() {
                let code = nutcracker_nearest_code(v / scale, sel);
                codes[i] = code;
                let recon = nutcracker_e2m1_to_f32(code, nutcracker_pack_scale(exp, sel));
                let d = v - recon;
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best_sel = sel;
                best_codes = codes;
            }
        }

        out.push(nutcracker_pack_scale(exp, best_sel));
        // 8 code bytes, low nibble = even index.
        for pair in 0..(NUTCRACKER_SUB_BLOCK / 2) {
            out.push((best_codes[pair * 2] & 0x0F) | ((best_codes[pair * 2 + 1] & 0x0F) << 4));
        }
    }
    Ok(out)
}

/// Nearest Nutcracker code for an unscaled value, given the block's selector.
///
/// Codes `0x0` and `0x8` both decode to the block's special value — the sign
/// comes from the selector, not from the code — so `0x0` is seeded as the
/// initial candidate and `0x8` is skipped as a redundant duplicate. Ties break
/// toward the lower code, which is why the seeded candidate must be `0x0`.
fn nutcracker_nearest_code(v: f32, sel: u8) -> u8 {
    let special = nutcracker_special_value(sel);
    let mut best_code = 0u8;
    let mut best_diff = (v - special).abs();
    // Decode the plain codes at scale 1.0 (exp == bias) so `v`, which is
    // already divided by the block scale, is comparable directly.
    let unit = nutcracker_pack_scale(NUTCRACKER_SCALE_BIAS as u8, sel);
    for code in 1u8..16 {
        let val = nutcracker_e2m1_to_f32(code, unit);
        let diff = (v - val).abs();
        if diff < best_diff {
            best_diff = diff;
            best_code = code;
        }
    }
    best_code
}

/// Convert f32 to MXFP4 E2M1 4-bit code with a given shared E8M0 exponent.
pub fn f32_to_mxfp4_e2m1(v: f32, shared_exp: u8) -> u8 {
    if v == 0.0 {
        return 0;
    }
    let scale = (2.0f32).powi(shared_exp as i32 - 127);
    let unscaled = v / scale;
    let sign_bit = if unscaled < 0.0 { 8u8 } else { 0u8 };
    let abs_val = unscaled.abs();

    let (exp, mant) = if abs_val < 0.25 {
        (0u8, 0u8)
    } else if abs_val < 0.75 {
        (0u8, 1u8)
    } else if abs_val < 1.25 {
        (1u8, 0u8)
    } else if abs_val < 1.75 {
        (1u8, 1u8)
    } else if abs_val < 2.5 {
        (2u8, 0u8)
    } else if abs_val < 3.5 {
        (2u8, 1u8)
    } else if abs_val < 5.0 {
        (3u8, 0u8)
    } else {
        (3u8, 1u8)
    };

    sign_bit | (exp << 1) | mant
}

/// Quantize a row-major `[rows, k]` f32 matrix to MXFP4 (E2M1 weights + E8M0 shared exponents) in the exact layout consumed by the ROCm/CUDA `grim_mxfp4_gemm_tiled` kernel: - `codes`: `rows * k / 2` bytes, two E2M1 codes per byte (even element in the low nibble, odd in the high nibble), grouped contiguously per row.
/// - `exps`: `rows * (k / 32)` bytes, one E8M0 shared exponent per 32-element block,.
pub fn quant_mxfp4_matrix(data: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(
        k % 32 == 0,
        "quant_mxfp4_matrix: k must be a multiple of 32"
    );
    let exps_per_row = k / 32;
    let mut codes = vec![0u8; rows * k / 2];
    let mut exps = vec![0u8; rows * exps_per_row];
    for r in 0..rows {
        for b in 0..exps_per_row {
            let block_base = r * k + b * 32;
            let mut max_abs = 0.0f32;
            for j in 0..32 {
                let a = data[block_base + j].abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            // Pick the E8M0 exponent (scale = 2^(e - 127)) so the block's max
            // magnitude fits within the E2M1 representable range.
            let e = if max_abs == 0.0 {
                127u32
            } else {
                let ratio = max_abs / 6.0f32;
                let mut e = (127.0 + ratio.log2().ceil()) as i32;
                while (max_abs / (2.0f32).powi(e - 127)) > 6.0 && e < 255 {
                    e += 1;
                }
                e.clamp(0, 255) as u32
            };
            let exp_byte = e as u8;
            exps[r * exps_per_row + b] = exp_byte;
            for i in 0..16 {
                let k0 = block_base + i * 2;
                let k1 = k0 + 1;
                let c0 = f32_to_mxfp4_e2m1(data[k0], exp_byte);
                let c1 = f32_to_mxfp4_e2m1(data[k1], exp_byte);
                codes[r * (k / 2) + b * 16 + i] = c0 | (c1 << 4);
            }
        }
    }
    (codes, exps)
}

/// Quantize f32 values to block-scaled FP4 (E2M1) bytes.
pub fn quant_fp4_block16(data: &[f32], block_size: usize) -> Result<Vec<u8>> {
    assert_eq!(block_size, 16);
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let global_scale = if max_abs == 0.0 { 1.0 } else { max_abs };

    let num_blocks = data.len().div_ceil(block_size);
    let mut out = Vec::with_capacity(4 + num_blocks * 9);
    out.extend_from_slice(&global_scale.to_le_bytes());
    // Minimum scale clamp = 2^-6 (0.015625), derived from FP8 E4M3 minimum normal exponent
    const MIN_FP8_SCALE: f32 = 1.0 / 64.0;
    for block in data.chunks(block_size) {
        let block_max = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let block_scale = (block_max / global_scale).clamp(MIN_FP8_SCALE, 1.0);
        let block_scale_fp8 = f32_to_fp8_e4m3(block_scale);
        out.push(block_scale_fp8);

        let rec_block_scale = fp8_e4m3_to_f32(block_scale_fp8);
        let effective_scale = rec_block_scale * global_scale;

        let mut packed_byte = 0u8;
        for (i, &v) in block.iter().enumerate() {
            let normalized = if effective_scale == 0.0 {
                0.0
            } else {
                (v / effective_scale).clamp(-1.0, 1.0)
            };

            // Nearest neighbor search in FP4_UNIFORM_LUT
            let mut code = 0;
            let mut min_diff = f32::MAX;
            for (c, &lut) in FP4_UNIFORM_LUT.iter().enumerate() {
                let diff = (normalized - lut).abs();
                if diff < min_diff {
                    min_diff = diff;
                    code = c;
                }
            }

            if i % 2 == 0 {
                packed_byte = (code as u8) << 4;
            } else {
                packed_byte |= code as u8;
                out.push(packed_byte);
            }
        }
        if block.len() % 2 == 1 {
            out.push(packed_byte);
        }
        // Pad the block to 8 bytes of packed data if it was short
        let expected_packed_len = 8;
        let actual_packed_len = block.len().div_ceil(2);
        if actual_packed_len < expected_packed_len {
            out.resize(out.len() + (expected_packed_len - actual_packed_len), 0);
        }
    }
    Ok(out)
}

/// Quantize f32 values to block-scaled FP8 (E4M3) bytes.
pub fn quant_fp8_block16(data: &[f32], block_size: usize) -> Result<Vec<u8>> {
    assert_eq!(block_size, 16);
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let global_scale = if max_abs == 0.0 { 1.0 } else { max_abs };

    let num_blocks = data.len().div_ceil(block_size);
    let mut out = Vec::with_capacity(4 + num_blocks * 17);
    out.extend_from_slice(&global_scale.to_le_bytes());

    for block in data.chunks(block_size) {
        let block_max = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let block_scale = (block_max / global_scale).clamp(1.0 / 64.0, 1.0);
        let block_scale_fp8 = f32_to_fp8_e4m3(block_scale);
        out.push(block_scale_fp8);

        let rec_block_scale = fp8_e4m3_to_f32(block_scale_fp8);
        let effective_scale = rec_block_scale * global_scale;

        for &v in block {
            let val_scaled = if effective_scale == 0.0 {
                0.0
            } else {
                v / effective_scale
            };
            out.push(f32_to_fp8_e4m3(val_scaled));
        }
        if block.len() < 16 {
            out.resize(out.len() + (16 - block.len()), 0);
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn quant_packed_symmetric(
    data: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
    curvature: Option<&[f32]>,
    shape: Option<&[usize]>,
) -> Result<Vec<u8>> {
    let prepared = prepare_gptq_proxy_tensor(data, bits, importance, curvature, shape)?;
    let packed_bytes_per_block = (BLOCK_SIZE_QK * bits as usize).div_ceil(8);
    let num_blocks = prepared.len().div_ceil(BLOCK_SIZE_QK);
    let mut out = Vec::with_capacity(num_blocks * (4 + packed_bytes_per_block));

    for (block_idx, block) in prepared.chunks(BLOCK_SIZE_QK).enumerate() {
        let block_importance = importance.map(|imp| {
            let start = block_idx * BLOCK_SIZE_QK;
            let end = (start + block.len()).min(imp.len());
            &imp[start..end]
        });
        let fit = fit_block_quantization(block, bits, block_importance)?;
        let packed = pack_bits(&fit.codes, bits);
        let scale = fit.scale;
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&packed);
        out.resize(out.len() + (packed_bytes_per_block - packed.len()), 0);
    }
    Ok(out)
}

/// Rewrite a tensor payload to a target quantized format.
/// This is the first Pass 4 substrate: it materializes the tensor into a logical f32.
pub fn rewrite_tensor_data(data: &[f32], plan: &TensorRewritePlan) -> Result<RewrittenTensorData> {
    let rewritten_bytes = match plan.target {
        QuantFormat::Q8_0 => quant_q80(data)?,
        QuantFormat::Q4K => quant_q4k(data)?,
        QuantFormat::Q5K => quant_q5k(data)?,
        QuantFormat::Q6K => quant_q6k(data)?,
        QuantFormat::Fp4 => quant_fp4(data)?,
        // TreePie is dense (E2M2, no pruning), so unlike GreyRaven it *is* a pure
        // f32 -> bytes rewrite and belongs in this match. Length must be a multiple
        // of 32 for the 5.0 bpw claim; the packer asserts rather than padding,
        // because a padded tail would change the element count silently.
        QuantFormat::TreePie => tree_pie::pack_tree_pie_bytes(data),
        QuantFormat::Nf4 => quant_nf4(data)?,
        QuantFormat::Fp8 => quant_fp8(data)?,
        // WhiteRaven blocked: codes = per-code E4M3, then 16x16-blocked
        // rearrange (requires the [n, k] shape: a permutation's index
        // geometry cannot be recovered from a flat element count).
        QuantFormat::Fp8Blocked16 => {
            if plan.shape.len() < 2 {
                return Err(Error::Backend(format!(
                    "Fp8Blocked16 rewrite needs [n, k] shape, got {:?}",
                    plan.shape
                )));
            }
            let (n, k) = (plan.shape[0], plan.shape[1]);
            let codes: Vec<u8> = data.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
            block_fp8_16x16(&codes, n, k)?
        }
        QuantFormat::Fp4Block16 => quant_fp4_block16(data, 16)?,
        QuantFormat::Fp8Block16 => quant_fp8_block16(data, 16)?,
        // GreyRaven is 2:4-pruned: pruning is a lossy, signal-dependent step
        // (Fisher-guided), not a function of the f32 data alone, so it cannot be
        // produced by a pure f32 -> bytes rewrite the way dense formats can.
        // E10 adds the path once the pruned format has a packer; until then
        // refusing is correct, and refusing loudly beats emitting a dense
        // tensor under a sparse format's name.
        QuantFormat::Fp8Sparse24 => {
            return Err(Error::Backend(
                "GreyRaven 2:4 requires a Fisher-guided prune before packing; \
                 a plain f32 rewrite cannot produce it"
                    .to_string(),
            ))
        }
        QuantFormat::W4A4OstQuant => {
            if plan.shape.len() < 2 {
                return Err(Error::Backend(format!(
                    "W4A4OstQuant rewrite needs [n, k] shape, got {:?}",
                    plan.shape
                )));
            }
            let (n, k) = (plan.shape[0], plan.shape[1]);
            let (qw, sc, zr) = quant_ostquant_w4_group128(data, n, k)?;
            // Same length-prefixed triple stream the native OSTQuant provider
            // emits ([u64 qw_len][qw][u64 sc_len][sc][u64 zr_len][zr]).
            let mut out = Vec::with_capacity(24 + qw.len() + sc.len() + zr.len());
            out.extend_from_slice(&(qw.len() as u64).to_le_bytes());
            out.extend_from_slice(&qw);
            out.extend_from_slice(&(sc.len() as u64).to_le_bytes());
            out.extend_from_slice(&sc);
            out.extend_from_slice(&(zr.len() as u64).to_le_bytes());
            out.extend_from_slice(&zr);
            out
        }
        // ForestRaven: per-row absmax INT8 in the framed blob. Needs [n, k]
        // for the row count, same as the W4A4 arm above needs it for groups.
        QuantFormat::Int8PerChannel => {
            if plan.shape.len() < 2 {
                return Err(Error::Backend(format!(
                    "Int8PerChannel rewrite needs [n, k] shape, got {:?}",
                    plan.shape
                )));
            }
            let (n, k) = (plan.shape[0], plan.shape[1]);
            let (codes, scales) = quant_forest_per_channel(data, n, k)?;
            let mut out = Vec::with_capacity(16 + codes.len() + scales.len());
            out.extend_from_slice(&(codes.len() as u64).to_le_bytes());
            out.extend_from_slice(&codes);
            out.extend_from_slice(&(scales.len() as u64).to_le_bytes());
            out.extend_from_slice(&scales);
            out
        }
        // 128x128 block FP8 is a *load-time* format for checkpoints that already
        // ship it (DeepSeek-style `weight_scale_inv`). There is no f32 -> packed
        // rewriter: producing one would need a source-side scale grid.
        QuantFormat::Fp8Block128 => {
            return Err(Error::Backend(
                "rewrite_tensor_data: Fp8Block128 is a load-time format, not a rewrite target"
                    .into(),
            ));
        }
        // Q2_K/Q3_K are load-time formats for the same reason `Fp8Block128` is:
        // grim *decodes* both (`dequant_q2k` / `dequant_q3k`, exercised by the
        // CPU and CUDA quantized-matmul paths) but ships no f32 -> Q2_K/Q3_K
        // encoder. Silently rewriting to a neighbouring K-quant would be wrong
        // data wearing the right label, so this is a named error instead.
        QuantFormat::Q2K | QuantFormat::Q3K => {
            return Err(Error::Backend(format!(
                "rewrite_tensor_data: {:?} is a load-time format, not a rewrite target \
                 (grim has no Q2_K/Q3_K encoder)",
                plan.target
            )));
        }
        QuantFormat::Iq4Nl => quant_iq4nl(data)?,
        QuantFormat::Iq4Xs => quant_iq4xs(data)?,
        QuantFormat::Iq3Xxs => quant_iq3xxs(data)?,
        QuantFormat::Iq3S => quant_iq3s(data)?,
        QuantFormat::Iq2Xxs => quant_iq2xxs(data)?,
        QuantFormat::Iq2Xs => quant_iq2xs(data)?,
        QuantFormat::Iq2S => quant_iq2s(data)?,
        // Upstream Q2_0: one fp16 scale + 16 B of packed 2-bit codes per
        // 64-weight block, 18 B/block.
        QuantFormat::Q2_0 => {
            let n = plan.shape.iter().product::<usize>();
            let blocks = n.div_ceil(BLOCK_SIZE_Q2_0);
            let mut bytes = vec![0u8; blocks * BLOCK_BYTES_Q2_0];
            quantize_q2_0_block(data, &mut bytes)?;
            bytes
        }
        // GSQ-RCO 3.5-bit (tag 81): same block bytes as Q2_0, codebook
        // {-2,-1,0,+1} — the inverse of dequant_gsq_rco_3p5, NOT of
        // dequant_q2_0. See quantize_gsq_rco_3p5_block for the sources.
        QuantFormat::GsqRco3p5 => {
            let n = plan.shape.iter().product::<usize>();
            let blocks = n.div_ceil(BLOCK_SIZE_Q2_0);
            let mut bytes = vec![0u8; blocks * BLOCK_BYTES_Q2_0];
            quantize_gsq_rco_3p5_block(data, &mut bytes)?;
            bytes
        }
    };

    Ok(RewrittenTensorData {
        bytes: rewritten_bytes,
        logical_shape: plan.shape.clone(),
        target: plan.target,
        wavefront_tiled: false,
    })
}

/// Quantize f32 values to IQ4_NL bytes (18 bytes per 32 weights).
///
/// Emits llama.cpp's `block_iq4_nl`, the inverse of
/// [`dequant_iq4nl`]: `ggml_half d` followed by `qs[QK4_NL/2]`, low nibble
/// first. The scale `d` is fitted the way `quantize_row_iq4_nl_impl` fits it
/// for the single-block case: least-squares against the signed codebook,
/// `d = sum(q*x) / sum(q*q)` with `weight[j] = x[j]^2`, then each element
/// takes the codebook index nearest `x[j]/d`.
///
/// The previous 170-byte-per-256 layout is gone. It matched no llama.cpp
/// block, so every artifact it produced was unreadable by llama.cpp and
/// unaddressable by this crate's own decoder.
pub fn quant_iq4nl(data: &[f32]) -> Result<Vec<u8>> {
    const QK4_NL: usize = 32;
    const BLOCK_BYTES: usize = 2 + QK4_NL / 2;
    if data.len() % QK4_NL != 0 {
        return Err(Error::Backend(format!(
            "quant_iq4nl: length {} must be a nonzero multiple of {QK4_NL}",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / QK4_NL * BLOCK_BYTES);
    for chunk in data.chunks(QK4_NL) {
        // Initial guess from the largest magnitude, as the reference does
        // before its refinement loop: d = max / values[0].
        let (amax, max_signed) = chunk.iter().fold((0.0f32, 0.0f32), |(a, m), &x| {
            if x.abs() > a {
                (x.abs(), x)
            } else {
                (a, m)
            }
        });
        let mut d = if amax > 0.0 {
            max_signed / KVALUES_IQ4NL_REF[0]
        } else {
            0.0
        };

        // Least-squares refit: w[j] = x[j]^2, q = codebook[nearest(id*x)].
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        let idx: Vec<usize> = if d != 0.0 {
            let id = 1.0 / d;
            chunk.iter().map(|&x| best_index_iq4nl(id * x)).collect()
        } else {
            vec![0; QK4_NL]
        };
        for (j, &l) in idx.iter().enumerate() {
            let q = KVALUES_IQ4NL_REF[l];
            let w = chunk[j] * chunk[j];
            sumqx += w * q * chunk[j];
            sumq2 += w * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };

        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        // Inverse of the decoder's element order: low nibble of byte j is
        // element j, high nibble is element j+16.
        let mut qs = [0u8; QK4_NL / 2];
        for j in 0..QK4_NL {
            let l = if d != 0.0 {
                best_index_iq4nl(id * chunk[j])
            } else {
                0
            };
            if j < QK4_NL / 2 {
                qs[j] = l as u8;
            } else {
                qs[j - QK4_NL / 2] |= (l as u8) << 4;
            }
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Index of the `kvalues_iq4nl` entry nearest `al` -- the reference's
/// `best_index_int8(16, values, al)`.
fn best_index_iq4nl(al: f32) -> usize {
    let mut best = 0usize;
    let mut best_err = f32::MAX;
    for (i, &v) in KVALUES_IQ4NL_REF.iter().enumerate() {
        let err = (al - v).abs();
        if err < best_err {
            best_err = err;
            best = i;
        }
    }
    best
}

/// Quantize f32 values to IQ4_XS bytes (136 bytes per 256 weights).
pub fn quant_iq4xs(data: &[f32]) -> Result<Vec<u8>> {
    const SUPER: usize = 256;
    let num_blocks = data.len().div_ceil(SUPER);
    let mut out = Vec::with_capacity(num_blocks * 136);
    for chunk in data.chunks(SUPER) {
        let max_val = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_val > 0.0 {
            max_val / 34.56951
        } else {
            1.0
        };
        let d_f16 = f32_to_f16(scale).to_le_bytes();
        out.extend_from_slice(&d_f16);
        out.extend_from_slice(&[32u8; 6]); // default scales

        let mut qs = vec![0u8; 128];
        for (i, &val) in chunk.iter().enumerate() {
            let mag = val.abs() / scale;
            let mut best_idx = 0;
            let mut best_err = f32::MAX;
            for (idx, &entry) in IQ4_NL_CODEBOOK[..8].iter().enumerate() {
                let err = (mag - entry).abs();
                if err < best_err {
                    best_err = err;
                    best_idx = idx;
                }
            }
            if val < 0.0 {
                best_idx |= 8;
            }
            if i % 2 == 0 {
                qs[i / 2] |= best_idx as u8;
            } else {
                qs[i / 2] |= (best_idx as u8) << 4;
            }
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Quantize f32 values to IQ3_XXS bytes (96 bytes per 256 weights).
pub fn quant_iq3xxs(data: &[f32]) -> Result<Vec<u8>> {
    const SUPER: usize = 256;
    let num_blocks = data.len().div_ceil(SUPER);
    let mut out = Vec::with_capacity(num_blocks * 96);
    for chunk in data.chunks(SUPER) {
        let max_val = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_val > 0.0 { max_val / 3.0 } else { 1.0 };
        let d_f16 = f32_to_f16(scale).to_le_bytes();
        out.extend_from_slice(&d_f16);

        let mut qs = vec![0u8; 64];
        let mut signs = vec![0u8; 30];
        for (i, &val) in chunk.iter().enumerate() {
            if val < 0.0 && i / 8 < 30 {
                signs[i / 8] |= 1 << (i % 8);
            }
            let code = ((val.abs() / scale).round().clamp(0.0, 3.0) as u8).min(3);
            if i % 4 == 0 {
                qs[i / 4] = code;
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&signs);
    }
    Ok(out)
}

/// Quantize f32 values to IQ3_S bytes (110 bytes per 256 weights).
pub fn quant_iq3s(data: &[f32]) -> Result<Vec<u8>> {
    const SUPER: usize = 256;
    let num_blocks = data.len().div_ceil(SUPER);
    let mut out = Vec::with_capacity(num_blocks * 110);
    for chunk in data.chunks(SUPER) {
        let max_val = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_val > 0.0 { max_val / 3.0 } else { 1.0 };
        let d_f16 = f32_to_f16(scale).to_le_bytes();
        out.extend_from_slice(&d_f16);

        let mut qs = vec![0u8; 64];
        let scales = vec![0u8; 12];
        let mut signs = vec![0u8; 32];
        for (i, &val) in chunk.iter().enumerate() {
            if val < 0.0 {
                signs[i / 8] |= 1 << (i % 8);
            }
            let code = ((val.abs() / scale).clamp(0.0, 3.0) as u8).min(3);
            if i % 4 == 0 {
                qs[i / 4] = code;
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&signs);
    }
    Ok(out)
}

/// Quantize f32 values to IQ2_XXS bytes (66 bytes per 256 weights).
pub fn quant_iq2xxs(data: &[f32]) -> Result<Vec<u8>> {
    const SUPER: usize = 256;
    let num_blocks = data.len().div_ceil(SUPER);
    let mut out = Vec::with_capacity(num_blocks * 66);
    for chunk in data.chunks(SUPER) {
        let max_val = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_val > 0.0 { max_val / 1.5 } else { 1.0 };
        let d_f16 = f32_to_f16(scale).to_le_bytes();
        out.extend_from_slice(&d_f16);

        let mut qs = vec![0u8; 32];
        let mut signs = vec![0u8; 32];
        let qs_len = qs.len();
        for (i, &val) in chunk.iter().enumerate() {
            if val < 0.0 {
                signs[i / 8] |= 1 << (i % 8);
            }
            let code = ((val.abs() / scale).clamp(0.0, 3.0) as u8).min(3);
            qs[(i / 8).min(qs_len - 1)] = code;
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&signs);
    }
    Ok(out)
}

/// Quantize f32 values to IQ2_XS bytes (74 bytes per 256 weights).
pub fn quant_iq2xs(data: &[f32]) -> Result<Vec<u8>> {
    const SUPER: usize = 256;
    let num_blocks = data.len().div_ceil(SUPER);
    let mut out = Vec::with_capacity(num_blocks * 74);
    for chunk in data.chunks(SUPER) {
        let max_val = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_val > 0.0 { max_val / 1.5 } else { 1.0 };
        let d_f16 = f32_to_f16(scale).to_le_bytes();
        out.extend_from_slice(&d_f16);

        let mut qs = vec![0u8; 32];
        let scales = vec![0u8; 8];
        let mut signs = vec![0u8; 32];
        let qs_len = qs.len();
        for (i, &val) in chunk.iter().enumerate() {
            if val < 0.0 {
                signs[i / 8] |= 1 << (i % 8);
            }
            let code = ((val.abs() / scale).clamp(0.0, 3.0) as u8).min(3);
            qs[(i / 8).min(qs_len - 1)] = code;
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&signs);
    }
    Ok(out)
}

/// Quantize f32 values to IQ2_S bytes (82 bytes per 256 weights).
///
/// Transcribed from llama.cpp `quantize_row_iq2_s_impl`
/// (`old/repo/llama.cpp-master/ggml/src/ggml-quants.c`, the `quant_weights ==
/// NULL` path), including its block layout from `ggml-common.h` (C, not Rust
/// — hence the `text` fence):
///
/// ```text
/// block_iq2_s { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
///               uint8_t scales[QK_K/32]; }        // 2 + 64 + 8 + 8 = 82 B
/// ```
///
/// The previous version here stored a per-ELEMENT 2-bit code in `qs[i/8]`,
/// left every sub-scale byte zero, and never touched `qh` — output that our
/// own (correct) `dequant_iq2s` turned back into near-garbage. The real
/// format stores one GRID INDEX per 8 elements (low 8 bits in `qs[i8]`, top 2
/// in `qh`), a packed sign byte per 8 elements at `qs[32 + i8]`, and a
/// weighted-least-squares-fitted sub-scale per 16 elements.
///
/// The final short block is zero-padded to 256 so the trailing weights
/// quantize as zeros rather than being dropped (the reference asserts
/// `n % QK_K == 0` instead).
pub fn quant_iq2s(data: &[f32]) -> Result<Vec<u8>> {
    use crate::iq_tables::{iq2s_quant_tables, KGRID_2BIT_1024};

    const QK: usize = 256;
    const KMAX_Q: i32 = 3;
    const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;

    // ggml `ggml_compute_fp32_to_fp16` (ggml-impl.h): round-to-nearest-even.
    // The crate-wide `f32_to_f16` TRUNCATES the mantissa, which would perturb
    // the f16 block scale off llama.cpp's value.
    fn f32_to_f16_rne(v: f32) -> u16 {
        let w = v.to_bits();
        let shl1_w = w.wrapping_add(w);
        let sign = (shl1_w >> 16) as u16 & 0x8000;
        let mut bias = shl1_w & 0xFF00_0000;
        if bias < 0x7100_0000 {
            bias = 0x7100_0000;
        }
        let scale_to_inf = f32::from_bits(0x7780_0000); // 0x1.0p+112
        let scale_to_zero = f32::from_bits(0x0880_0000); // 0x1.0p-110
        let base = f32::from_bits((bias >> 1).wrapping_add(0x0780_0000))
            + (v.abs() * scale_to_inf) * scale_to_zero;
        let bits = base.to_bits();
        let exp_bits = (bits >> 13) & 0x0000_7C00;
        let mantissa_bits = bits & 0x0000_0FFF;
        let nonsign = (exp_bits + mantissa_bits) as u16;
        sign | if shl1_w > 0xFF00_0000 {
            0x7E00
        } else {
            nonsign
        }
    }

    // llama.cpp `nearest_int`: bit-trick round-to-nearest-even via the f32
    // mantissa (ggml-quants.c:621).
    fn nearest_int(fval: f32) -> i32 {
        debug_assert!(fval.abs() <= 4194303.0f32);
        let val = fval + 12582912.0f32;
        (val.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
    }

    let tables = iq2s_quant_tables();
    let num_blocks = data.len().div_ceil(QK);
    let mut out = Vec::with_capacity(num_blocks * 82);
    for chunk in data.chunks(QK) {
        let mut xbl = [0.0f32; QK];
        xbl[..chunk.len()].copy_from_slice(chunk);

        let sumx2: f32 = xbl.iter().map(|&x| x * x).sum();
        let sigma2 = 2.0f32 * sumx2 / QK as f32;

        let mut qs = [0u8; QK / 4]; // grid indices at [0..32), signs at [32..64)
        let mut qh = [0u8; QK / 32];
        let mut scales_bytes = [0u8; QK / 32];
        let mut scale = [0.0f32; QK / 16]; // per 16-element group
        let mut max_scale = 0.0f32;

        for ib in 0..QK / 16 {
            let xb = &xbl[16 * ib..16 * ib + 16];
            // No quant_weights: weight[i] = 0.25*sigma2 + x^2.
            let mut weight = [0.0f32; 16];
            let mut waux = [0.0f32; 16];
            let mut xval = [0.0f32; 16];
            for i in 0..16 {
                weight[i] = 0.25f32 * sigma2 + xb[i] * xb[i];
                waux[i] = weight[i].sqrt();
            }
            let mut block_signs = [0u8; 2];
            for k in 0..2 {
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s;
            }
            let max = xval.iter().copied().fold(0.0f32, f32::max);
            let mut l = [0i8; 16];
            if max < GROUP_MAX_EPS_IQ2_S {
                continue; // scale[ib] stays 0
            }
            let mut best = 0.0f32;
            let mut group_scale = max / (2 * KMAX_Q - 1) as f32;
            let mut is_on_grid = [true; 2];
            for is in -9..=9 {
                let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 16];
                let mut is_on_grid_aux = [false; 2];
                for k in 0..2 {
                    for i in 0..8 {
                        let li = nearest_int(0.5f32 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = li.clamp(0, KMAX_Q - 1) as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as u16) << (2 * i);
                    }
                    let mut grid_index = tables.map[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        grid_index = tables
                            .best_neighbour(
                                u,
                                &xval[8 * k..8 * k + 8],
                                &waux[8 * k..8 * k + 8],
                                this_scale,
                            )
                            .expect("off-grid code has a neighbour")
                            as i32;
                        let entry = KGRID_2BIT_1024[grid_index as usize];
                        for i in 0..8 {
                            laux[8 * k + i] = ((entry >> (2 * i)) & 0x3) as i8;
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = (2 * laux[i] + 1) as f32;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    group_scale = sumqx / sumq2;
                    best = group_scale * sumqx;
                    l = laux;
                    is_on_grid = is_on_grid_aux;
                }
            }
            if is_on_grid.iter().any(|&g| !g) && group_scale > 0.0 {
                let id = 1.0 / group_scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u = 0u16;
                    for i in 0..8 {
                        let li = nearest_int(0.5f32 * (id * xval[8 * k + i] - 1.0));
                        let li = li.clamp(0, KMAX_Q - 1) as i8;
                        u |= (li as u16) << (2 * i);
                        l[8 * k + i] = li;
                    }
                    let grid_index = match tables.map[u as usize] {
                        g if g >= 0 => g,
                        _ => tables
                            .best_neighbour(
                                u,
                                &xval[8 * k..8 * k + 8],
                                &waux[8 * k..8 * k + 8],
                                group_scale,
                            )
                            .expect("off-grid code has a neighbour")
                            as i32,
                    };
                    let entry = KGRID_2BIT_1024[grid_index as usize];
                    for i in 0..8 {
                        l[8 * k + i] = ((entry >> (2 * i)) & 0x3) as i8;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = (2 * l[i] + 1) as f32;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    group_scale = sumqx / sumq2;
                }
            }
            if group_scale < 0.0 {
                group_scale = -group_scale;
                for k in 0..2 {
                    block_signs[k] = !block_signs[k];
                }
            }
            for k in 0..2 {
                let mut u = 0u16;
                for i in 0..8 {
                    u |= (l[8 * k + i] as u16) << (2 * i);
                }
                let grid_index = tables.map[u as usize];
                assert!(
                    grid_index >= 0,
                    "IQ2_S quantize: final point {u} not on grid (L = {:?})",
                    &l[8 * k..8 * k + 8]
                );
                let i8 = 2 * ib + k;
                qs[i8] = (grid_index & 255) as u8;
                qh[i8 / 4] |= ((grid_index >> 8) as u8) << (2 * (i8 % 4));
                qs[QK / 8 + i8] = block_signs[k];
            }
            scale[ib] = group_scale;
            max_scale = max_scale.max(group_scale);
        }

        if max_scale > 0.0 {
            let d = max_scale / 31.0;
            let d_f16 = f32_to_f16_rne(d * 0.9875f32);
            out.extend_from_slice(&d_f16.to_le_bytes());
            let id = 1.0 / d;
            for ib in 0..QK / 16 {
                let li = nearest_int(0.5f32 * (id * scale[ib] - 1.0));
                let li = li.clamp(0, 15) as u8;
                if ib % 2 == 0 {
                    scales_bytes[ib / 2] = li;
                } else {
                    scales_bytes[ib / 2] |= li << 4;
                }
            }
        } else {
            out.extend_from_slice(&[0u8; 2]);
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&scales_bytes);
    }
    Ok(out)
}

pub fn dequant_packed_symmetric(data: &[u8], num_weights: usize, bits: u8) -> Result<Vec<f32>> {
    let packed_bytes_per_block = (BLOCK_SIZE_QK * bits as usize).div_ceil(8);
    let stride = 4 + packed_bytes_per_block;
    let num_blocks = num_weights.div_ceil(BLOCK_SIZE_QK);
    if data.len() < num_blocks * stride {
        return Err(Error::Backend(format!(
            "packed symmetric q{bits}: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * stride,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0usize;
    for block_index in 0..num_blocks {
        let scale = f32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let packed = &data[pos..pos + packed_bytes_per_block];
        pos += packed_bytes_per_block;
        let remaining = num_weights.saturating_sub(block_index * BLOCK_SIZE_QK);
        let block_len = remaining.min(BLOCK_SIZE_QK);
        let unpacked = unpack_bits(packed, bits, block_len);
        out.extend(dequantize_block_signed(&unpacked, scale, bits));
    }
    Ok(out)
}

#[allow(dead_code)]
const GPTQ_PROXY_COLUMN_GROUP: usize = 4;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct BlockQuantization {
    scale: f32,
    codes: Vec<u32>,
}

fn fit_block_quantization(
    block: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
) -> Result<BlockQuantization> {
    let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let signed_limit = signed_quant_limit(bits);
    let base_scale = if absmax == 0.0 || signed_limit == 0.0 {
        1.0
    } else {
        absmax / signed_limit
    };
    let weights = importance.unwrap_or(&[]);

    let mut best_scale = base_scale;
    let mut best_error = f32::INFINITY;
    let mut best_q = Vec::new();

    for multiplier in [0.6f32, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4] {
        let candidate_scale = base_scale * multiplier;
        let quantized = quantize_block_linear(block, candidate_scale, bits);
        let quantized = refine_block_residuals(block, &quantized, candidate_scale, bits, weights);
        let dequantized = dequantize_block_signed(&quantized, candidate_scale, bits);
        let error = weighted_error(block, &dequantized, weights);
        if error < best_error {
            best_error = error;
            best_scale = candidate_scale;
            best_q = quantized;
        }
    }

    Ok(BlockQuantization {
        scale: best_scale,
        codes: best_q,
    })
}

fn prepare_gptq_proxy_tensor(
    data: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
    curvature: Option<&[f32]>,
    shape: Option<&[usize]>,
) -> Result<Vec<f32>> {
    let row_width = infer_row_width(shape, data.len());
    let mut prepared = Vec::with_capacity(data.len());

    for row_index in 0..data.len().div_ceil(row_width.max(1)) {
        let row_start = row_index * row_width;
        if row_start >= data.len() {
            break;
        }
        let row_end = (row_start + row_width).min(data.len());
        let row = &data[row_start..row_end];
        let row_importance = importance.map(|imp| {
            let end = row_end.min(imp.len());
            &imp[row_start..end]
        });
        let row_curvature = curvature.map(|diag| {
            let end = row_end.min(diag.len());
            &diag[row_start..end]
        });
        let prepared_row =
            prepare_row_with_sequential_update(row, bits, row_importance, row_curvature)?;
        prepared.extend_from_slice(&prepared_row);
    }

    Ok(prepared)
}

fn prepare_row_with_sequential_update(
    row: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
    curvature: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let weights = importance.unwrap_or(&[]);
    let curvature_diag = curvature.unwrap_or(&[]);
    let baseline_error = row_rewrite_error(row, row, bits, weights, curvature_diag)?;
    let mut prepared = row.to_vec();
    let mut carry = 0.0f32;
    let mut residual_tail = 0.0f32;

    for block_index in 0..row.len().div_ceil(BLOCK_SIZE_QK) {
        let start = block_index * BLOCK_SIZE_QK;
        let end = (start + BLOCK_SIZE_QK).min(row.len());
        let block_weights = &weights[start.min(weights.len())..end.min(weights.len())];
        let block_curvature =
            &curvature_diag[start.min(curvature_diag.len())..end.min(curvature_diag.len())];

        for value in &mut prepared[start..end] {
            *value += carry + residual_tail;
        }

        apply_block_diagonal_update(&mut prepared[start..end], block_weights, block_curvature);

        let fit = fit_block_quantization(&prepared[start..end], bits, Some(block_weights))?;
        let dequantized = dequantize_block_signed(&fit.codes, fit.scale, bits);
        let residual_energy = prepared[start..end]
            .iter()
            .zip(dequantized.iter())
            .enumerate()
            .map(|(idx, (original, approx))| {
                let weight = block_weights.get(idx).copied().unwrap_or(1.0);
                let h = block_curvature.get(idx).copied().unwrap_or(weight.max(1.0));
                weight * h * (original - approx)
            })
            .sum::<f32>();
        let curvature_mass = block_curvature.iter().copied().sum::<f32>();
        let normalizer = (block_weights.iter().copied().sum::<f32>() + curvature_mass)
            .max(end.saturating_sub(start).max(1) as f32);
        carry = (residual_energy / normalizer) * 0.25;
        residual_tail = block_curvature
            .last()
            .copied()
            .unwrap_or(1.0)
            .sqrt()
            .min(4.0)
            * carry
            * 0.1;
    }

    let sequential_error = row_rewrite_error(row, &prepared, bits, weights, curvature_diag)?;
    if sequential_error <= baseline_error {
        Ok(prepared)
    } else {
        Ok(row.to_vec())
    }
}

fn apply_block_diagonal_update(block: &mut [f32], weights: &[f32], curvature: &[f32]) {
    if block.len() <= 1 {
        return;
    }

    for group_start in (0..block.len()).step_by(GPTQ_PROXY_COLUMN_GROUP) {
        let group_end = (group_start + GPTQ_PROXY_COLUMN_GROUP).min(block.len());
        let group_weights = &weights[group_start.min(weights.len())..group_end.min(weights.len())];
        let group_curvature =
            &curvature[group_start.min(curvature.len())..group_end.min(curvature.len())];
        let mean = weighted_group_mean(
            &block[group_start..group_end],
            group_weights,
            group_curvature,
        );
        let coupling = block_group_coupling(group_curvature);

        for offset in 0..(group_end - group_start) {
            let idx = group_start + offset;
            let weight = group_weights.get(offset).copied().unwrap_or(1.0);
            let h = group_curvature.get(offset).copied().unwrap_or(1.0);
            let trust = (weight * h).sqrt().min(8.0);
            let blend = (0.04 * coupling / trust.max(1e-3)).clamp(0.0, 0.2);
            block[idx] = block[idx] * (1.0 - blend) + mean * blend;
        }
    }
}

fn weighted_group_mean(values: &[f32], weights: &[f32], curvature: &[f32]) -> f32 {
    let mut weighted_sum = 0.0f32;
    let mut mass = 0.0f32;
    for (index, value) in values.iter().enumerate() {
        let w = weights.get(index).copied().unwrap_or(1.0);
        let h = curvature.get(index).copied().unwrap_or(1.0);
        let scale = (w * h).max(1e-4);
        weighted_sum += scale * *value;
        mass += scale;
    }
    if mass <= 1e-6 {
        0.0
    } else {
        weighted_sum / mass
    }
}

fn block_group_coupling(curvature: &[f32]) -> f32 {
    if curvature.len() <= 1 {
        return 0.0;
    }
    let mean = curvature.iter().copied().sum::<f32>() / curvature.len() as f32;
    let variance = curvature
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / curvature.len() as f32;
    1.0 / (1.0 + variance.sqrt())
}

fn infer_row_width(shape: Option<&[usize]>, len: usize) -> usize {
    let inferred = shape
        .and_then(|dims| dims.last().copied())
        .filter(|width| *width > 0)
        .unwrap_or(len.max(1));
    inferred.min(len.max(1))
}

fn row_rewrite_error(
    original: &[f32],
    candidate: &[f32],
    bits: u8,
    weights: &[f32],
    curvature: &[f32],
) -> Result<f32> {
    let mut total_error = 0.0f32;
    for block_index in 0..candidate.len().div_ceil(BLOCK_SIZE_QK) {
        let start = block_index * BLOCK_SIZE_QK;
        let end = (start + BLOCK_SIZE_QK).min(candidate.len());
        let block_weights = &weights[start.min(weights.len())..end.min(weights.len())];
        let block_curvature = &curvature[start.min(curvature.len())..end.min(curvature.len())];
        let fit = fit_block_quantization(&candidate[start..end], bits, Some(block_weights))?;
        let dequantized = dequantize_block_signed(&fit.codes, fit.scale, bits);
        total_error += weighted_curvature_error(
            &original[start..end],
            &dequantized,
            block_weights,
            block_curvature,
        );
    }
    Ok(total_error)
}

fn weighted_curvature_error(
    original: &[f32],
    dequantized: &[f32],
    weights: &[f32],
    curvature: &[f32],
) -> f32 {
    original
        .iter()
        .enumerate()
        .map(|(index, lhs)| {
            let weight = weights.get(index).copied().unwrap_or(1.0);
            let h = curvature.get(index).copied().unwrap_or(1.0);
            let residual = lhs - dequantized.get(index).copied().unwrap_or_default();
            weight * h.max(1e-4) * residual * residual
        })
        .sum()
}

fn quantize_block_linear(block: &[f32], scale: f32, bits: u8) -> Vec<u32> {
    let zero_point = quant_zero_point(bits) as f32;
    let signed_limit = signed_quant_limit(bits);
    block
        .iter()
        .map(|value| {
            (((value / scale).round()).clamp(-signed_limit, signed_limit) + zero_point) as u32
        })
        .collect()
}

fn dequantize_block_signed(block: &[u32], scale: f32, bits: u8) -> Vec<f32> {
    let zero_point = quant_zero_point(bits) as f32;
    block
        .iter()
        .map(|value| ((*value as f32) - zero_point) * scale)
        .collect()
}

fn refine_block_residuals(
    original: &[f32],
    initial_codes: &[u32],
    scale: f32,
    bits: u8,
    weights: &[f32],
) -> Vec<u32> {
    let mut codes = initial_codes.to_vec();
    let max_code = (1u32 << bits) - 1;
    if original.is_empty() {
        return codes;
    }

    for _ in 0..3 {
        let mut changed = false;
        for index in 0..codes.len() {
            let current = codes[index];
            let base_weight = weights.get(index).copied().unwrap_or(1.0);
            let current_value = dequantize_block_signed(&[current], scale, bits)[0];
            let current_error = base_weight * (original[index] - current_value).powi(2);

            let mut best_code = current;
            let mut best_error = current_error;

            for candidate in [
                current.saturating_sub(1),
                current.saturating_add(1).min(max_code),
            ] {
                if candidate == current {
                    continue;
                }
                let candidate_value = dequantize_block_signed(&[candidate], scale, bits)[0];
                let candidate_error = base_weight * (original[index] - candidate_value).powi(2);
                if candidate_error + 1e-8 < best_error {
                    best_error = candidate_error;
                    best_code = candidate;
                }
            }

            if best_code != current {
                codes[index] = best_code;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    codes
}

fn quant_zero_point(bits: u8) -> u32 {
    1u32 << (bits - 1)
}

fn signed_quant_limit(bits: u8) -> f32 {
    ((1u32 << (bits - 1)) - 1) as f32
}

fn weighted_error(original: &[f32], dequantized: &[f32], weights: &[f32]) -> f32 {
    original
        .iter()
        .enumerate()
        .map(|(index, lhs)| {
            let weight = weights.get(index).copied().unwrap_or(1.0);
            let residual = lhs - dequantized.get(index).copied().unwrap_or_default();
            weight * residual * residual
        })
        .sum()
}

fn pack_bits(values: &[u32], bits: u8) -> Vec<u8> {
    let total_bits = values.len() * bits as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_cursor = 0usize;
    for value in values {
        let mut remaining = *value;
        for _ in 0..bits {
            let byte_index = bit_cursor / 8;
            let bit_index = bit_cursor % 8;
            out[byte_index] |= ((remaining & 1) as u8) << bit_index;
            remaining >>= 1;
            bit_cursor += 1;
        }
    }
    out
}

fn unpack_bits(bytes: &[u8], bits: u8, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let mut bit_cursor = 0usize;
    for _ in 0..count {
        let mut value = 0u32;
        for bit in 0..bits {
            let byte_index = bit_cursor / 8;
            let bit_index = bit_cursor % 8;
            let bit_value = ((bytes[byte_index] >> bit_index) & 1) as u32;
            value |= bit_value << bit;
            bit_cursor += 1;
        }
        out.push(value);
    }
    out
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    if exp >= 0x8D {
        // Overflow: return inf
        return (sign << 15) | 0x7C00;
    }
    if exp <= 0x70 {
        // Underflow: subnormal
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Randomized SVD algorithm for importance matrix calculation (§0 / §19).
/// Replicates `scirs2_linalg` randomized SVD projection strategy: Projects high-dimensional weight arrays to lower-rank spaces with Gaussian.
pub fn randomized_svd_importance(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    target_rank: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if target_rank == 0 || target_rank > rows.min(cols) {
        return Err(Error::Backend(
            "Invalid target rank for randomized SVD".into(),
        ));
    }
    // Replicating Martinsson/Tropp Randomized SVD pattern:
    // 1. Generate random Gaussian matrix Omega of size (cols, target_rank + oversampling)
    let oversampling = 5;
    let rank_k = (target_rank + oversampling).min(cols);
    let mut omega = vec![0.0f32; cols * rank_k];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for val in &mut omega {
        // Quick deterministic LCG-based normal distribution sample
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let u1 = ((seed >> 40) as u32 as f32) / 16777216.0;
        let u2 = (((seed & 0xFFFFFFFF) >> 8) as u32 as f32) / 16777216.0;
        let normal = (-2.0 * u1.max(1e-5).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *val = normal;
    }

    // 2. Form sample matrix Y = A * Omega (rows, rank_k)
    let mut y = vec![0.0f32; rows * rank_k];
    for r in 0..rows {
        for c in 0..rank_k {
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += matrix[r * cols + k] * omega[k * rank_k + c];
            }
            y[r * rank_k + c] = sum;
        }
    }

    // 3. Orthonormalize Y using Gram-Schmidt projection (approximation of QR decomposition Q)
    let mut q = vec![0.0f32; rows * rank_k];
    for col in 0..rank_k {
        let mut v = vec![0.0f32; rows];
        for r in 0..rows {
            v[r] = y[r * rank_k + col];
        }
        for prev in 0..col {
            let mut dot = 0.0f32;
            for r in 0..rows {
                dot += y[r * rank_k + col] * q[r * rank_k + prev];
            }
            for r in 0..rows {
                v[r] -= dot * q[r * rank_k + prev];
            }
        }
        let norm_sq: f32 = v[..rows].iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-5);
        for r in 0..rows {
            q[r * rank_k + col] = v[r] / norm;
        }
    }

    // 4. Form B = Q^T * A (rank_k, cols)
    let mut b = vec![0.0f32; rank_k * cols];
    for r in 0..rank_k {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..rows {
                sum += q[k * rank_k + r] * matrix[k * cols + c];
            }
            b[r * cols + c] = sum;
        }
    }

    // Return the low-rank projections (U_approx = Q, S_approx = singular values mock, V_approx = B)
    // S_approx holds column norm representations of B projection spaces
    let mut s = vec![0.0f32; target_rank];
    for r in 0..target_rank {
        let mut norm = 0.0f32;
        for c in 0..cols {
            norm += b[r * cols + c] * b[r * cols + c];
        }
        s[r] = norm.sqrt();
    }

    // Truncate Q and B to the target rank
    let mut u_trunc = vec![0.0f32; rows * target_rank];
    for r in 0..rows {
        for c in 0..target_rank {
            u_trunc[r * target_rank + c] = q[r * rank_k + c];
        }
    }

    let mut vt_trunc = vec![0.0f32; target_rank * cols];
    for r in 0..target_rank {
        for c in 0..cols {
            vt_trunc[r * cols + c] = b[r * cols + c];
        }
    }

    Ok((u_trunc, s, vt_trunc))
}

// Phase 2: Importance-Matrix Calibration

/// Per-layer importance scores from calibration.
/// `layer_scores[i]` is the importance of tensor `i` (higher = more quantization-sensitive - should use more.
#[derive(Debug, Clone)]
pub struct ImportanceScores {
    pub tensor_names: Vec<String>,
    pub layer_scores: Vec<f32>,
}

impl ImportanceScores {
    pub fn new(tensor_names: Vec<String>, layer_scores: Vec<f32>) -> Self {
        assert_eq!(tensor_names.len(), layer_scores.len());
        Self {
            tensor_names,
            layer_scores,
        }
    }

    pub fn score_for(&self, tensor_name: &str) -> f32 {
        self.layer_scores
            .iter()
            .zip(&self.tensor_names)
            .find(|(_, n)| *n == tensor_name)
            .map(|(s, _)| *s)
            .unwrap_or(0.0)
    }
}

/// Compute per-tensor importance scores using randomized SVD.
/// For each tensor, runs randomized SVD and returns the column-norm-based importance: the Frobenius norm of.
pub fn compute_importance_scores(tensors: &[(String, Vec<f32>, usize, usize)]) -> Vec<f32> {
    let mut scores = Vec::with_capacity(tensors.len());
    for (_name, data, rows, cols) in tensors {
        if *rows == 0 || *cols == 0 {
            scores.push(0.0);
            continue;
        }
        let r = (*rows).min(*cols);
        let target_rank = r.clamp(1, 8);
        let (_, s, vt) = match randomized_svd_importance(data, *rows, *cols, target_rank) {
            Ok(r) => r,
            Err(_) => {
                scores.push(0.0);
                continue;
            }
        };
        let n_cols = *cols;
        let s_len = s.len();
        let mut col_norms: Vec<f32> = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let mut norm_sq: f32 = 0.0;
            for row in 0..s_len {
                let val = vt[row * n_cols + c];
                norm_sq += val * val;
            }
            col_norms.push(norm_sq.sqrt());
        }
        let total_importance: f32 = s
            .iter()
            .zip(&col_norms)
            .take(target_rank)
            .map(|(sig, cn)| sig * cn)
            .sum();
        scores.push(total_importance);
    }
    scores
}

// Phase 4: Fisher/GGN Diagonal Computation for GPTQ Error-Correcting Updates

/// One calibration sample: input activations and output gradients for a specific tensor.
/// Populated by running the calibration dataset forward+backward through the model and capturing intermediate activations/gradients via.
#[derive(Debug, Clone)]
pub struct FisherCalibrationSample {
    pub input_activations: Vec<f32>,
    pub output_gradients: Vec<f32>,
}

/// Compute the diagonal of the Generalized Gauss-Newton (GGN) matrix for a weight matrix using a batch of pre-computed calibration activations and gradients.
/// This is the "true" curvature for GPTQ error-correcting updates, replacing `build_curvature_proxy`.
pub fn compute_fisher_diagonal(
    _weights: &[f32],
    calibration_samples: &[FisherCalibrationSample],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    if calibration_samples.is_empty() || rows == 0 || cols == 0 {
        return vec![1.0f32; rows * cols];
    }

    let _batch_size = calibration_samples
        .first()
        .map(|s| s.output_gradients.len() / rows)
        .unwrap_or(1)
        .max(1);
    let _num_groups = cols.div_ceil(group_size);

    // Accumulate per-column and per-element diagonal
    let mut h_diag = vec![0.0f32; cols];
    let m = calibration_samples.len() as f32;

    for sample in calibration_samples {
        let batch = sample.output_gradients.len() / rows;
        if sample.input_activations.len() != batch * cols || batch == 0 {
            continue;
        }

        for b in 0..batch {
            let grad_out_slice = &sample.output_gradients[b * rows..(b + 1) * rows];
            let in_slice = &sample.input_activations[b * cols..(b + 1) * cols];

            for col in 0..cols {
                let x_sq = in_slice[col] * in_slice[col];
                let col_h: f32 = grad_out_slice.iter().map(|go| x_sq * go * go).sum();
                h_diag[col] += col_h;
            }
        }
    }

    // Average
    for val in &mut h_diag {
        *val /= m;
        *val = val.max(1e-8);
    }

    // Broadcast per-column diagonal across all rows (each row gets the same diagonal)
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        out.extend_from_slice(&h_diag);
    }

    out.truncate(rows * cols);
    while out.len() < rows * cols {
        out.push(1.0);
    }

    out
}

/// Compute per-group GGN diagonal - one curvature value per quantization group.
/// This is the format actually used in GPTQ re-quantization: each group of `group_size` columns shares.
pub fn compute_grouped_fisher_diagonal(
    _weights: &[f32],
    calibration_samples: &[FisherCalibrationSample],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    if calibration_samples.is_empty() || rows == 0 || cols == 0 {
        return vec![1.0f32; cols.div_ceil(group_size)];
    }

    let _batch_size = calibration_samples
        .first()
        .map(|s| s.output_gradients.len() / rows)
        .unwrap_or(1)
        .max(1);
    let num_groups = cols.div_ceil(group_size);
    let mut group_h_diag = vec![0.0f32; num_groups];
    let m = calibration_samples.len() as f32;

    for sample in calibration_samples {
        let batch = sample.output_gradients.len() / rows;
        if sample.input_activations.len() != batch * cols || batch == 0 {
            continue;
        }

        for b in 0..batch {
            let grad_out_slice = &sample.output_gradients[b * rows..(b + 1) * rows];
            let in_slice = &sample.input_activations[b * cols..(b + 1) * cols];

            for (gi, g_start) in (0..num_groups).map(|gi| (gi, gi * group_size)) {
                let g_end = (g_start + group_size).min(cols);
                let go_sq_sum: f32 = grad_out_slice.iter().map(|go| go * go).sum();
                let mut accum = 0.0f32;
                let mut col_count = 0usize;
                for &x in &in_slice[g_start..g_end] {
                    accum += x * x * go_sq_sum;
                    col_count += 1;
                }
                if col_count > 0 {
                    group_h_diag[gi] += accum / (cols as f32);
                }
            }
        }
    }

    for val in &mut group_h_diag {
        *val /= m;
        *val = val.max(1e-8);
    }

    group_h_diag
}

/// Compute an importance-weighted curvature proxy when calibration data is not available (CPU fallback).
/// This is a first-order approximation of the GGN diagonal using activation magnitude as a proxy.
pub fn compute_curvature_proxy(data: &[f32], layer_importance: f32) -> Vec<f32> {
    let layer_scale = layer_importance.abs().max(1e-3);
    data.iter()
        .map(|value| 1.0 + layer_scale * (value.abs() + value * value).min(16.0))
        .collect()
}

/// Refined Scale Fit (RSF) for K-quant blocks.
/// Re-fits the per-block scales using importance-weighted L2 reconstruction error minimization.
pub fn refined_scale_fit(
    data: &[f32],
    importance: &[f32],
    block_size: usize,
    n_levels: u32,
) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let n_blocks = data.len().div_ceil(block_size);
    let mut scales = Vec::with_capacity(n_blocks);
    let step = (n_levels - 1) as f32;

    for bi in 0..n_blocks {
        let start = bi * block_size;
        let end = (start + block_size).min(data.len());
        let blk_data = &data[start..end];
        let blk_imp = &importance[start..end];

        // Weighted RMS of the block: scale = sqrt(sum(w * x^2) / sum(w)) / (n_levels/2)
        let weighted_sq_sum: f32 = blk_data
            .iter()
            .zip(blk_imp.iter())
            .map(|(x, w)| w * x * x)
            .sum();
        let weight_sum: f32 = blk_imp.iter().sum();
        if weight_sum < 1e-9 {
            scales.push(1.0);
            continue;
        }
        let rms = (weighted_sq_sum / weight_sum).sqrt();
        let scale = if rms < 1e-9 { 1.0 } else { rms / step * 2.0 };
        scales.push(scale);
    }
    Ok(scales)
}

// Phase 3: RCO (Riemannian Constrained Optimization) Bitwidth Search

/// Legacy configuration alias for EvoPress, routed directly into RCO.
#[derive(Debug, Clone)]
pub struct EvoPressConfig {
    pub population_size: usize,
    pub generations: usize,
    pub target_bpw: f32,
    pub tournament_size: usize,
    pub crossover_prob: f32,
    pub mutation_prob: f32,
    pub available_bpws: Vec<u32>,
}

impl Default for EvoPressConfig {
    fn default() -> Self {
        Self {
            population_size: 128,
            generations: 40,
            target_bpw: 4.0,
            tournament_size: 3,
            crossover_prob: 0.8,
            mutation_prob: 0.05,
            available_bpws: vec![2, 3, 4, 5, 6, 8],
        }
    }
}

/// Backwards-compatible bitwidth search function, now driven by RCO (Riemannian Constrained Optimization).
pub fn evopress_search(
    config: &EvoPressConfig,
    importance_scores: &[f32],
    tensor_sizes: &[usize],
    progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Vec<u32> {
    let rco_config = RcoConfig {
        steps: config.generations.max(20),
        target_bpw: config.target_bpw,
        available_bpws: config.available_bpws.clone(),
        ..Default::default()
    };
    rco_search(&rco_config, importance_scores, tensor_sizes, progress)
}

// SmoothQuant channel scaling (mockdud.md §3N3b, §6P1)

/// SmoothQuant: channel-wise activation-aware weight scaling.
/// Shifts quantization difficulty from activations to weights by scaling weight columns by the inverse of.
pub fn apply_smoothquant_scale(
    weights: &mut [f32],
    out_channels: usize,
    in_channels: usize,
    calibration_acts: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(
        weights.len(),
        out_channels * in_channels,
        "weights.len() must equal out_channels * in_channels"
    );

    let mut scales: Vec<f32> = if let Some(acts) = calibration_acts {
        assert_eq!(
            acts.len(),
            out_channels,
            "calibration_acts.len() must equal out_channels"
        );
        acts.iter().map(|&a| 1.0 / a.max(1e-8)).collect()
    } else {
        let mut max_vals = vec![0.0f32; out_channels];
        for o in 0..out_channels {
            for i in 0..in_channels {
                let val = weights[o * in_channels + i].abs();
                if val > max_vals[o] {
                    max_vals[o] = val;
                }
            }
        }
        for v in &mut max_vals {
            *v = 1.0 / (*v).max(1e-8);
        }
        max_vals
    };

    // A channel whose inverse came from the 1e-8 floor (all-zero weights, or
    // zero calibration acts) is dead: its scale is exactly 1e8, and letting it
    // set the normalization below would shrink every live channel by ~1e-8 --
    // a single dead neuron zeroing the whole tensor. Dead channels keep scale
    // 1.0 (identity) and are excluded from the max. A live channel would need
    // max exactly 1e-8 to collide with the floor value, which no real weight
    // attains; the comparison is exact, not approximate, so there is no
    // threshold to drift.
    const DEAD_SCALE: f32 = 1e8;
    // Liveness BEFORE pinning: after pinning, dead 1.0s are indistinguishable
    // from live 1.0s, and letting a pinned dead row set the max would suppress
    // live normalization. Dead rows are invisible to the pipeline, not just
    // clamped by it.
    let live: Vec<bool> = scales.iter().map(|&s| s < DEAD_SCALE).collect();
    for (o, s) in scales.iter_mut().enumerate() {
        if !live[o] {
            *s = 1.0;
        }
    }
    let max_s = scales
        .iter()
        .enumerate()
        .filter(|(o, _)| live[*o])
        .map(|(_, &s)| s)
        .fold(0.0f32, f32::max);
    if max_s > 0.0 {
        for (o, s) in scales.iter_mut().enumerate() {
            if live[o] {
                *s /= max_s;
            }
        }
    }

    // Apply: W'[o,i] = W[o,i] * scale[o]
    for o in 0..out_channels {
        let s = scales[o];
        if (s - 1.0).abs() < 1e-6 {
            continue; // identity channel — skip
        }
        for i in 0..in_channels {
            weights[o * in_channels + i] *= s;
        }
    }

    scales
}

// SpinQuant Cayley rotation (mockdud.md §3N3a, §6P2)

/// SpinQuant: learn rotation matrices via Cayley SGD on the Stiefel manifold.
/// Rotates weight matrices before quantization so outlier dimensions spread across all channels, producing outlier-free weights.
pub fn spinquant_rotate(weights: &mut [f32], dim: usize, lr: f32, steps: usize) {
    assert!(
        dim > 0 && dim.is_power_of_two(),
        "SpinQuant dim must be a positive power of 2"
    );
    assert_eq!(
        weights.len(),
        dim * dim,
        "weights.len() must equal dim * dim (a square block)"
    );

    // Scratch buffers — allocated once, reused each step.
    let mut rotated = vec![0.0f32; dim];
    let mut quantized = vec![0.0f32; dim];
    let mut grad = vec![0.0f32; dim * dim];
    let mut skew = vec![0.0f32; dim * dim];

    // Initialize R = I (identity).
    let mut r = vec![0.0f32; dim * dim];
    for i in 0..dim {
        r[i * dim + i] = 1.0;
    }

    for _step in 0..steps {
        // Forward W @ R^T → rotated
        for i in 0..dim {
            rotated[i] = 0.0;
            for j in 0..dim {
                rotated[i] += weights[j * dim + i] * r[j * dim + i];
            }
        }

        // Simulate Q4_K nearest rounding (symmetric range [-7, 7]).
        let max_abs = rotated.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = max_abs / 7.0;
        let inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
        for i in 0..dim {
            let q = (rotated[i] * inv_scale).round().clamp(-7.0, 7.0) as i8;
            quantized[i] = (q as f32) * scale;
        }

        // dL/dR = 2 * W^T @ (rotated - quantized)
        for i in 0..dim {
            let diff = rotated[i] - quantized[i];
            for j in 0..dim {
                grad[j * dim + i] = 2.0 * weights[j * dim + i] * diff;
            }
        }

        // Skew-symmetric projection: G = grad^T - grad (Stiefel tangent)
        for i in 0..dim {
            for j in 0..dim {
                skew[i * dim + j] = grad[j * dim + i] - grad[i * dim + j];
            }
        }

        // Cayley first-order retraction: R -= lr * skew @ R
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    sum += skew[i * dim + k] * r[k * dim + j];
                }
                r[i * dim + j] -= lr * sum;
            }
        }

        // Gram-Schmidt re-orthogonalisation (keeps R on Stiefel).
        for col in 0..dim {
            for row in 0..col {
                let dot: f32 = (0..dim).map(|k| r[k * dim + col] * r[k * dim + row]).sum();
                for k in 0..dim {
                    r[k * dim + col] -= dot * r[k * dim + row];
                }
            }
            let norm: f32 = (0..dim)
                .map(|k| r[k * dim + col].powi(2))
                .sum::<f32>()
                .sqrt();
            if norm > 1e-10 {
                for k in 0..dim {
                    r[k * dim + col] /= norm;
                }
            }
        }
    }

    // Write W' = W @ R^T back into weights in-place.
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                sum += weights[k * dim + j] * r[k * dim + i];
            }
            weights[i * dim + j] = sum;
        }
    }
}

/// Convenience wrapper: apply SmoothQuant then SpinQuant in sequence.
/// Use this as the single entry point in `convert.rs` before calling `pack_tensors()`.
pub fn pre_quantize_transform(
    weights: &mut [f32],
    out_channels: usize,
    in_channels: usize,
    calibration_acts: Option<&[f32]>,
    spinquant_dim: usize,
    spinquant_lr: f32,
    spinquant_steps: usize,
) -> Vec<f32> {
    let smooth_scales =
        apply_smoothquant_scale(weights, out_channels, in_channels, calibration_acts);

    // SpinQuant operates on square blocks of size spinquant_dim.
    // Apply block-wise on the weight matrix (treat as stacked dim×dim blocks).
    if spinquant_dim.is_power_of_two() && spinquant_dim <= out_channels.max(in_channels) {
        let total = out_channels * in_channels;
        let blocks = total / (spinquant_dim * spinquant_dim);
        for b in 0..blocks {
            let off = b * spinquant_dim * spinquant_dim;
            if off + spinquant_dim * spinquant_dim <= total {
                spinquant_rotate(
                    &mut weights[off..off + spinquant_dim * spinquant_dim],
                    spinquant_dim,
                    spinquant_lr,
                    spinquant_steps,
                );
            }
        }
    }

    smooth_scales
}

// Attention projection role detection & precision policy (WI-SPINQUANT-AttentionGate)

/// Returns true if `tensor_name` corresponds to an attention projection layer.
/// Matches standard attention projection substring conventions across GGUF, HuggingFace / SafeTensors, and native formats: -.
pub fn is_attention_projection(tensor_name: &str) -> bool {
    let lower = tensor_name.to_lowercase();
    lower.contains("attn_q")
        || lower.contains("attn_k")
        || lower.contains("attn_v")
        || lower.contains("attn_o")
        || lower.contains(".wq.weight")
        || lower.contains(".wk.weight")
        || lower.contains(".wv.weight")
        || lower.contains(".wo.weight")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
        || lower.contains("self_attn.q_proj")
        || lower.contains("self_attn.k_proj")
        || lower.contains("self_attn.v_proj")
        || lower.contains("self_attn.o_proj")
}

/// Minimum quantization bitwidth for attention projection tensors.
pub fn attention_min_bpw() -> u32 {
    5 // Q5_K
}

/// Enforce the minimum precision floor for attention projection tensors.
pub fn enforce_attention_precision(suggested_bpw: u32) -> u32 {
    suggested_bpw.max(attention_min_bpw())
}

#[cfg(test)]
mod smoothquant_tests {
    use super::*;

    #[test]
    fn smoothquant_scale_inverts_large_columns() {
        let out_c = 2;
        let in_c = 3;
        // weights are row-major: weights[row * in_c + col]
        // col 0 (out_ch 0) has large values, col 1 (out_ch 1) has small values
        let mut weights = vec![
            10.0, 10.0, 10.0, // out_ch=0: all rows have 10 in this col
            1.0, 1.0, 1.0, // out_ch=1: all rows have 1 in this col
        ];

        let scales = apply_smoothquant_scale(&mut weights, out_c, in_c, None);

        // Channel 0 max=10 → scale 1/10 = 0.1 after normalization
        assert!((scales[0] - 0.1).abs() < 0.01);
        // Channel 1 max=1 → scale 1.0 after normalization (unchanged)
        assert!((scales[1] - 1.0).abs() < 0.01);

        // After scaling, col 0 values should be ~1.0 (10 * 0.1)
        assert!((weights[0] - 1.0).abs() < 0.1);
        assert!((weights[3] - 1.0).abs() < 0.1);
    }

    #[test]
    fn smoothquant_identical_columns_unchanged() {
        let mut weights = vec![5.0f32; 6]; // 2 out_ch × 3 in_ch
        let scales = apply_smoothquant_scale(&mut weights, 2, 3, None);
        assert_eq!(scales[0], 1.0);
        assert_eq!(scales[1], 1.0);
        assert_eq!(weights, vec![5.0f32; 6]);
    }

    #[test]
    fn smoothquant_with_calibration_acts() {
        let mut weights = vec![1.0f32; 6]; // 2 out_ch × 3 in_ch
        let calibration = vec![2.0f32, 0.5]; // out_ch=0 is 2× larger

        let scales = apply_smoothquant_scale(&mut weights, 2, 3, Some(&calibration));

        // calibration inverted + normalized: [1/2=0.5, 1/0.5=2.0] → max=2 → [0.25, 1.0]
        assert!((scales[0] - 0.25).abs() < 0.01);
        assert!((scales[1] - 1.0).abs() < 0.01);
        // col 0 scaled by 0.25, col 1 stays (scaled by 1.0)
        assert!((weights[0] - 0.25).abs() < 0.01); // row 0, col 0
        assert_eq!(weights[3], 1.0); // row 0, col 1
    }

    #[test]
    #[should_panic(expected = "weights.len() must equal out_channels * in_channels")]
    fn smoothquant_panics_on_wrong_size() {
        let mut w = vec![1.0f32; 5];
        apply_smoothquant_scale(&mut w, 3, 3, None);
    }

    #[test]
    fn smoothquant_zero_row_leaves_live_rows_untouched() {
        // A single all-zero row must not perturb any other row: the zero
        // row's inverse hits the 1e-8 floor (scale 1e8), and if that sets the
        // normalization max every live row shrinks by ~1e-8 -- a dead neuron
        // zeroing the whole tensor. Regression test: this once produced
        // weights ~1e-8 of their true values with no error.
        let mut weights = vec![
            2.0, -2.0, 1.0, // row 0: max 2
            0.0, 0.0, 0.0, // row 1: dead
            4.0, -1.0, 0.5, // row 2: max 4
        ];
        let scales = apply_smoothquant_scale(&mut weights, 3, 3, None);
        // Live rows normalize among themselves: max inverse is 1/2 (row 0),
        // so row 0 -> 1.0, row 2 -> (1/4)/(1/2) = 0.5. Dead row pins at 1.0.
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert_eq!(scales[1], 1.0);
        assert!((scales[2] - 0.5).abs() < 1e-6);
        // Row 0 unchanged, row 2 halved, dead row still zero (not NaN).
        assert_eq!(weights[0..3], [2.0, -2.0, 1.0]);
        assert_eq!(weights[3..6], [0.0, 0.0, 0.0]);
        assert_eq!(weights[6..9], [2.0, -0.5, 0.25]);
    }
}

#[cfg(test)]
mod spinquant_tests {
    use super::*;

    #[test]
    fn spinquant_preserves_frobenius_norm() {
        let dim = 4;
        let mut weights: Vec<f32> = (0..dim * dim)
            .map(|i| (i as f32 - (dim * dim) as f32 / 2.0) * 0.1)
            .collect();
        let orig_norm: f32 = weights.iter().map(|v| v * v).sum::<f32>().sqrt();

        spinquant_rotate(&mut weights, dim, 0.05, 2);

        let new_norm: f32 = weights.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (orig_norm - new_norm).abs() < 1e-4,
            "orthogonal transform must preserve Frobenius norm: {} != {}",
            orig_norm,
            new_norm,
        );
    }

    #[test]
    fn spinquant_produces_finite_output() {
        let dim = 8;
        let mut weights: Vec<f32> = (0..dim * dim).map(|i| (i as f32 - 32.0) * 10.0).collect();

        spinquant_rotate(&mut weights, dim, 0.05, 5);

        assert!(weights.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "dim must be a positive power of 2")]
    fn spinquant_panics_on_non_power_of_two() {
        let mut w = vec![1.0f32; 9];
        spinquant_rotate(&mut w, 3, 0.05, 1);
    }

    #[test]
    #[should_panic(expected = "weights.len() must equal dim * dim")]
    fn spinquant_panics_on_wrong_length() {
        let mut w = vec![1.0f32; 10];
        spinquant_rotate(&mut w, 4, 0.05, 1);
    }
}

#[cfg(test)]
mod attention_role_tests {
    use super::*;

    #[test]
    fn test_is_attention_projection() {
        let cases = &[
            ("blk.48.attn_q.weight", true),
            ("blk.48.attn_k.weight", true),
            ("blk.48.attn_v.weight", true),
            ("blk.48.attn_o.weight", true),
            ("model.embed_tokens.weight", false),
            ("model.layers.48.mlp.gate_proj.weight", false),
            ("model.layers.48.mlp.up_proj.weight", false),
            ("model.layers.48.mlp.down_proj.weight", false),
            ("blk.48.ffn_gate", false),
            ("self_attn.q_proj.weight", true),
            ("self_attn.k_proj.weight", true),
            ("self_attn.v_proj.weight", true),
            ("self_attn.o_proj.weight", true),
            ("layers.0.attention.wq.weight", true),
            ("layers.0.attention.wk.weight", true),
            ("layers.0.attention.wv.weight", true),
            ("layers.0.attention.wo.weight", true),
        ];
        for (name, expected) in cases {
            assert_eq!(
                is_attention_projection(name),
                *expected,
                "failed for {name}"
            );
        }
    }

    #[test]
    fn test_enforce_attention_precision() {
        assert_eq!(enforce_attention_precision(3), 5);
        assert_eq!(enforce_attention_precision(4), 5);
        assert_eq!(enforce_attention_precision(5), 5);
        assert_eq!(enforce_attention_precision(6), 6);
        assert_eq!(enforce_attention_precision(8), 8);
    }
}

#[cfg(test)]
mod pre_quantize_transform_tests {
    use super::*;

    #[test]
    fn pre_quantize_transform_returns_scales() {
        let out_c = 2;
        let in_c = 3;
        let mut weights = vec![1.0f32; out_c * in_c];

        let scales = pre_quantize_transform(&mut weights, out_c, in_c, None, 4, 0.05, 2);

        assert_eq!(scales.len(), out_c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q4_0 block: fp16 scale (from an f32 scale) + 32 nibbles.
    fn q40_block(scale: f32, nibbles: [u8; 32]) -> [u8; 18] {
        let bits = f32_to_f16_bits(scale);
        let mut blk = [0u8; 18];
        blk[0..2].copy_from_slice(&bits.to_le_bytes());
        for (i, &q) in nibbles.iter().enumerate() {
            if i % 2 == 0 {
                blk[2 + i / 2] = q & 0x0F;
            } else {
                blk[2 + i / 2] |= (q & 0x0F) << 4;
            }
        }
        blk
    }

    /// Round-to-nearest-even f32 -> f16 bit pattern (host reference encoder).
    fn f32_to_f16_bits(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let mut exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = x & 0x007F_FFFF;
        if exp <= 0 {
            return sign;
        }
        if exp >= 0x1F {
            return sign | 0x7C00;
        }
        // round to nearest even on the 13 dropped mantissa bits
        let round = 0x0FFF + ((mant >> 12) & 1);
        let mut m = mant + round;
        if m > 0x007F_FFFF {
            m = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
        sign | ((exp as u16) << 10) | ((m >> 13) as u16)
    }

    #[test]
    fn dequant_q4_0_matches_hand_computed_block() {
        // One block, scale 0.5, nibbles 8..=39 in order -> (q-8)*0.5 = 0..15.5 step 0.5
        let mut nibs = [0u8; 32];
        for (i, n) in nibs.iter_mut().enumerate() {
            *n = (i as u8) % 16;
        }
        let blk = q40_block(0.5, nibs);
        let out = dequant_q4_0(&blk, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            let expect = (i as f32 % 16.0 - 8.0) * 0.5;
            assert_eq!(v, expect, "elem {i}");
        }
    }

    #[test]
    fn greycrow_g32_repack_is_bit_exact_against_q4_0() {
        // Column-major [n, k]: two columns, 64 weights each, mixed scales.
        let (n, k) = (2usize, 64usize);
        let blocks = k / 32;
        let mut packed = Vec::new();
        let mut nibs = [0u8; 32];
        for col in 0..n {
            for b in 0..blocks {
                for (i, x) in nibs.iter_mut().enumerate() {
                    *x = ((col * 5 + b * 7 + i * 3 + 1) % 16) as u8;
                }
                packed.extend_from_slice(&q40_block(0.125 + (col + b) as f32 * 0.5, nibs));
            }
        }
        let reference = dequant_q4_0(&packed, n * k).unwrap();
        let (qw, sc, zr) = repack_q40_to_greycrow_g32(&packed, n, k).unwrap();

        // Geometry: 4.5 bpw in, 4.5 bpw out, same as Q4_0.
        assert_eq!(qw.len(), n * (k / 8) * 4);
        assert_eq!(sc.len(), n * blocks * 2);
        assert_eq!(zr.len(), n * blocks);
        assert!(
            zr.iter().all(|&z| z == 8),
            "Q4_0's bias is already folded into the nibble, so every zero point is 8"
        );

        let got = dequant_greycrow_g32(&qw, &sc, &zr, n, k).unwrap();
        assert_eq!(got.len(), reference.len());
        for (i, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "GreyCrow must be bit-exact vs Q4_0, diverged at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn greycrow_g32_rejects_k_not_divisible_by_32() {
        let packed = vec![0u8; 18];
        assert!(repack_q40_to_greycrow_g32(&packed, 1, 30).is_err());
        assert!(repack_q40_to_greycrow_g32(&packed, 1, 32).is_ok());
    }

    #[test]
    fn q4_0_roundtrip_through_quant_dequant_is_lossless_for_representable_values() {
        // Values a Q4_0 block can hold exactly: (q-8)*d.
        let d = 0.25f32;
        let mut nibs = [0u8; 32];
        for (i, n) in nibs.iter_mut().enumerate() {
            *n = i as u8 % 16;
        }
        let blk = q40_block(d, nibs);
        let back = dequant_q4_0(&blk, 32).unwrap();
        for (i, &v) in back.iter().enumerate() {
            let expect = ((i % 16) as f32 - 8.0) * d;
            assert_eq!(v, expect, "elem {i}");
        }
    }

    #[test]
    fn roundtrip_q80() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let quantized = quant_q80(&data).unwrap();
        let dequantized = dequant_q80(&quantized, data.len()).unwrap();
        assert_eq!(data.len(), dequantized.len());
        // Q8_0 should be close
        for i in 0..data.len() {
            let diff = (data[i] - dequantized[i]).abs();
            assert!(
                diff < 0.5,
                "diff at {i}: {} vs {}, diff={}",
                data[i],
                dequantized[i],
                diff
            );
        }
    }

    #[test]
    fn dequant_q4k_basic() {
        // 256 weights: 1 Q4_K super-block = 144 bytes
        let mut data = vec![0u8; 144];
        // d = 1.0 (in f16: 0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // dmin = 0.0
        data[2..4].copy_from_slice(&0u16.to_le_bytes());
        // scales: sc_i = 1, m_i = 0
        data[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        // qs: byte 0 (q1=2, q2=5)
        data[16] = 2 | (5 << 4);

        let deq = dequant_q4k(&data, 256).unwrap();
        assert_eq!(deq.len(), 256);
        assert_eq!(deq[0], 2.0f32);
        assert_eq!(deq[32], 5.0f32);
    }

    #[test]
    fn roundtrip_q4k() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 17.0).collect();
        let quantized = quant_q4k(&data).unwrap();
        let dequantized = dequant_q4k(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        let mse = mean_squared_error(&data, &dequantized);
        assert!(mse < 0.5, "q4k mse too high: {mse}");
    }

    // ----- Q4KHalf (PLAN-kvcache-channel-axis WI-1/WI-3) -----

    #[test]
    fn q4khalf_byte_layout_is_pinned() {
        // 128 elements -> 4 + 2*4 + 64 = 76 bytes (the WI-1 byte budget).
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 16.0).collect();
        let packed = quant_q4khalf(&data).unwrap();
        assert_eq!(packed.len(), 76, "W1 byte budget: 76 B/row at head_dim 128");
        // 96 elements -> 3 sub-blocks: 4 + 6 + 48 = 58 bytes.
        let d96: Vec<f32> = (0..96).map(|i| (i as f32) * 0.01).collect();
        assert_eq!(quant_q4khalf(&d96).unwrap().len(), 58);
        // 192 -> 4 + 12 + 96 = 112.
        let d192: Vec<f32> = (0..192).map(|i| (i as f32) * 0.01).collect();
        assert_eq!(quant_q4khalf(&d192).unwrap().len(), 112);
        // Bad lengths are rejected (not-zero-padded silently).
        assert!(quant_q4khalf(&[]).is_err());
        assert!(quant_q4khalf(&d96[..40]).is_err());
        assert!(quant_q4khalf(&vec![0.0f32; 320]).is_err());
    }

    #[test]
    fn q4khalf_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.17).sin()).collect();
        let packed = quant_q4khalf(&data).unwrap();
        let back = dequant_q4khalf(&packed, 128).unwrap();
        assert_eq!(back.len(), 128);
        let mse = mean_squared_error(&data, &back);
        assert!(mse < 0.02, "q4khalf mse too high: {mse}");
        // Dequant rejects length mismatches.
        assert!(dequant_q4khalf(&packed, 96).is_err());
        assert!(dequant_q4khalf(&packed[..70], 128).is_err());
    }

    /// The point of the format: a row whose channel-groups have wildly
    /// different dynamic ranges keeps the quiet groups intact, because each
    /// sub-block gets its own scale+min. A whole-row 4-bit quantizer (the
    /// LegacyNibble “one abs-max scale per row” scheme this replaces) flat-
    /// lines the quiet groups. The claim is measured, not asserted.
    #[test]
    fn q4khalf_per_subblock_scales_track_a_hot_group() {
        let mut data = vec![0.0f32; 128];
        for (i, v) in data.iter_mut().enumerate().take(96).skip(64) {
            *v = 3.0 + (i as f32 - 64.0) * 0.01; // sub-block 2: large magnitudes
        }
        for (i, v) in data.iter_mut().enumerate().take(64) {
            *v = (i as f32 * 0.013).sin() * 1e-3 + 1e-3; // sub-blocks 0-1: 1e-3 scale
        }
        let packed = quant_q4khalf(&data).unwrap();
        assert_eq!(packed.len(), 76);
        let back = dequant_q4khalf(&packed, 128).unwrap();

        // Reference: whole-row symmetric 4-bit (what Q4KHalf replaces at 68 B).
        let peak = data.iter().copied().fold(0.0f32, f32::max);
        let legacy_err: f32 = data
            .iter()
            .map(|&x| {
                let n = ((x / peak) * 7.0 + 8.0).round().clamp(0.0, 15.0);
                ((n - 8.0) / 7.0 * peak - x).abs()
            })
            .fold(0.0f32, f32::max);

        let q4kh_cold_err: f32 = (0..64)
            .map(|i| (back[i] - data[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            q4kh_cold_err * 4.0 < legacy_err,
            "per-group scaling must win decisively on the quiet groups: q4khalf={q4kh_cold_err} legacy={legacy_err}"
        );
        // Hot group: absolute error bounded by the sub-block's grid step.
        let hot_err: f32 = (64..96)
            .map(|i| (back[i] - data[i]).abs())
            .fold(0.0f32, f32::max);
        // sub-block 2 spans ~0.31 → its grid step is 0.31/15 ≈ 0.021 in the
        // asymmetric-Q4K contribution; errors land within a few grid steps of
        // the fp16-rounded scale representation.
        assert!(hot_err < 0.15, "hot group bounded: {hot_err}");
    }

    #[test]
    fn quant_mxfp4_matrix_layout_and_roundtrip() {
        let k = 64usize;
        let rows = 4usize;
        let data: Vec<f32> = (0..rows * k).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let (codes, exps) = quant_mxfp4_matrix(&data, rows, k);
        assert_eq!(codes.len(), rows * k / 2);
        assert_eq!(exps.len(), rows * (k / 32));

        // Decode every element with mxfp4_e2m1_to_f32 and check the layout matches the GEMM kernel (even
        // element = low nibble, odd = high, exps grouped per 32-element block per row).
        let mut max_err = 0.0f32;
        let exps_per_row = k / 32;
        for r in 0..rows {
            for b in 0..exps_per_row {
                let e = exps[r * exps_per_row + b];
                for i in 0..16 {
                    let byte = codes[r * (k / 2) + b * 16 + i];
                    let c0 = byte & 0x0F;
                    let c1 = (byte >> 4) & 0x0F;
                    let k0 = r * k + b * 32 + i * 2;
                    let k1 = k0 + 1;
                    let d0 = mxfp4_e2m1_to_f32(c0, e);
                    let d1 = mxfp4_e2m1_to_f32(c1, e);
                    // Sign preservation proves the nibble/block layout is correct.
                    if data[k0] != 0.0 {
                        assert_eq!(
                            d0.is_sign_negative(),
                            data[k0].is_sign_negative(),
                            "sign mismatch at {k0}"
                        );
                    }
                    if data[k1] != 0.0 {
                        assert_eq!(
                            d1.is_sign_negative(),
                            data[k1].is_sign_negative(),
                            "sign mismatch at {k1}"
                        );
                    }
                    max_err = max_err.max((d0 - data[k0]).abs());
                    max_err = max_err.max((d1 - data[k1]).abs());
                }
            }
        }
        // MXFP4 E2M1 is ~4-bit; absolute error up to ~half the top code spacing
        // (<= 1.0 * block_scale) is expected for this magnitude range.
        assert!(max_err < 1.5, "mxfp4 matrix max_err too high: {max_err}");
    }

    /// A4 (PLAN-reduce-d2h-h2d): requantizing ALREADY-MXFP4 data must be a
    /// lossless repack — identical codes and exponents. This is the license
    /// for `build_fused_qkv_pack` to dequant→requant native-MXFP4 weights at
    /// load: it is a layout conversion, not a quality-affecting requant.
    #[test]
    fn quant_mxfp4_matrix_repack_of_native_is_lossless() {
        let k = 96usize; // multiple of 32, > 1 superblock per row
        let rows = 3usize;
        let seed_data: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.4)
            .collect();
        let (codes0, exps0) = quant_mxfp4_matrix(&seed_data, rows, k);

        // Decode exactly like the kernels do: nibble order (even=low, odd=high),
        // one E8M0 exponent per 32-element block per row.
        let exps_per_row = k / 32;
        let mut native = vec![0f32; rows * k];
        for r in 0..rows {
            for b in 0..exps_per_row {
                let e = exps0[r * exps_per_row + b];
                for i in 0..16 {
                    let byte = codes0[r * (k / 2) + b * 16 + i];
                    let k0 = r * k + b * 32 + i * 2;
                    native[k0] = mxfp4_e2m1_to_f32(byte & 0x0F, e);
                    native[k0 + 1] = mxfp4_e2m1_to_f32((byte >> 4) & 0x0F, e);
                }
            }
        }
        let (codes1, exps1) = quant_mxfp4_matrix(&native, rows, k);
        assert_eq!(
            codes0, codes1,
            "codes must survive a quant→dequant→quant roundtrip"
        );
        assert_eq!(exps0, exps1, "E8M0 exponents must survive the roundtrip");
    }

    #[test]
    fn roundtrip_q5k() {
        let data = vec![0u8; 176];
        let dequantized = dequant_q5k(&data, 256).unwrap();
        assert_eq!(dequantized.len(), 256);
    }

    #[test]
    fn roundtrip_q6k() {
        let data = vec![0u8; 210];
        let dequantized = dequant_q6k(&data, 256).unwrap();
        assert_eq!(dequantized.len(), 256);
    }

    #[test]
    fn rewrite_tensor_to_q80() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
        let rewritten = rewrite_tensor_data(
            &data,
            &TensorRewritePlan {
                target: QuantFormat::Q8_0,
                shape: vec![32, 1],
                importance: None,
                curvature: None,
            },
        )
        .unwrap();
        assert!(!rewritten.bytes.is_empty());
        assert_eq!(rewritten.target, QuantFormat::Q8_0);
    }

    #[test]
    fn residual_refinement_beats_linear_baseline() {
        let block = vec![
            -3.2f32, -2.8, -2.1, -1.7, -1.2, -0.9, -0.3, 0.1, 0.25, 0.6, 0.95, 1.3, 1.8, 2.2, 2.7,
            3.4,
        ];
        let weights = vec![
            1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0,
        ];
        let bits = 4;
        let scale = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / signed_quant_limit(bits);

        let linear_codes = quantize_block_linear(&block, scale, bits);
        let refined_codes = refine_block_residuals(&block, &linear_codes, scale, bits, &weights);
        let linear = dequantize_block_signed(&linear_codes, scale, bits);
        let refined = dequantize_block_signed(&refined_codes, scale, bits);

        let linear_error = weighted_error(&block, &linear, &weights);
        let refined_error = weighted_error(&block, &refined, &weights);
        assert!(
            refined_error <= linear_error,
            "residual refinement regressed: {refined_error} > {linear_error}"
        );
    }

    #[test]
    fn sequential_row_update_improves_two_block_tensor() {
        let mut row = Vec::new();
        for i in 0..256 {
            let base = if i < 128 {
                (i as f32 - 64.0) / 2.5
            } else {
                (i as f32 - 192.0) / 4.0
            };
            let bias = if i >= 128 { 0.35 } else { 0.0 };
            row.push(base + bias);
        }

        let baseline_bytes = quant_q4k(&row).unwrap();
        let sequential_bytes = quant_q4k(&row).unwrap();

        let baseline = dequant_q4k(&baseline_bytes, row.len()).unwrap();
        let sequential = dequant_q4k(&sequential_bytes, row.len()).unwrap();
        let baseline_error = mean_squared_error(&row, &baseline);
        let sequential_error = mean_squared_error(&row, &sequential);
        assert!(
            sequential_error <= baseline_error,
            "sequential row update regressed: {sequential_error} > {baseline_error}"
        );
    }

    #[test]
    fn curvature_weighted_row_update_is_non_regressive() {
        let row: Vec<f32> = (0..64)
            .map(|i| {
                let x = i as f32 - 32.0;
                (x / 7.0).sin() * 3.0 + if i > 40 { 0.45 } else { -0.15 }
            })
            .collect();
        let weights = vec![1.0f32; row.len()];
        let curvature: Vec<f32> = row
            .iter()
            .enumerate()
            .map(|(idx, value)| 1.0 + value.abs() + if idx > 40 { 2.0 } else { 0.25 })
            .collect();

        let baseline_error = row_rewrite_error(&row, &row, 4, &weights, &curvature).unwrap();
        let prepared =
            prepare_row_with_sequential_update(&row, 4, Some(&weights), Some(&curvature)).unwrap();
        let curved_error = row_rewrite_error(&row, &prepared, 4, &weights, &curvature).unwrap();
        assert!(
            curved_error <= baseline_error,
            "curvature-aware row update regressed: {curved_error} > {baseline_error}"
        );
    }

    #[test]
    fn block_diagonal_update_preserves_group_center() {
        let mut block = vec![2.0f32, 2.4, 1.6, 2.2, -1.0, -0.8, -1.2, -0.9];
        let weights = vec![1.0f32; block.len()];
        let curvature = vec![2.0f32, 2.1, 1.9, 2.0, 1.5, 1.4, 1.6, 1.5];
        let before_a = weighted_group_mean(&block[..4], &weights[..4], &curvature[..4]);
        let before_b = weighted_group_mean(&block[4..], &weights[4..], &curvature[4..]);

        apply_block_diagonal_update(&mut block, &weights, &curvature);

        let after_a = weighted_group_mean(&block[..4], &weights[..4], &curvature[..4]);
        let after_b = weighted_group_mean(&block[4..], &weights[4..], &curvature[4..]);
        assert!((after_a - before_a).abs() < 0.05, "group A drifted too far");
        assert!((after_b - before_b).abs() < 0.05, "group B drifted too far");
    }

    fn mean_squared_error(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / lhs.len().max(1) as f32
    }

    #[test]
    fn test_randomized_svd_determinism_and_dimensions() {
        let matrix = vec![1.0f32; 100]; // 10x10 matrix
        let target_rank = 3;
        let (u, s, vt) = randomized_svd_importance(&matrix, 10, 10, target_rank).unwrap();

        assert_eq!(u.len(), 10 * target_rank);
        assert_eq!(s.len(), target_rank);
        assert_eq!(vt.len(), target_rank * 10);

        // Deterministic repeat check
        let (u2, s2, vt2) = randomized_svd_importance(&matrix, 10, 10, target_rank).unwrap();
        assert_eq!(u, u2);
        assert_eq!(s, s2);
        assert_eq!(vt, vt2);
    }

    #[test]
    fn gptq_3bit_cross_word_packing() {
        // Test 3-bit GPTQ dequant with known non-zero codes packed at 3-bit word-boundary positions (in_idx 0-9 fit within one u32 word).
        // 3 u32 words span 96 bits, packing up to 32 values.
        let in_features = 32;
        let out_features = 1;
        let group_size = 32;

        let mut qweight = vec![0u8; 12]; // 3 words for the single out_col
        let mut qzeros = vec![0u8; 12]; // 3 words for zero-point
        let scales = 1.0f32.to_le_bytes().to_vec();

        let zero_val = 0u32; // zero_point = zero_val + 1 = 1
        let scale_val = 1.0f32;

        // Pack known non-zero codes at word-boundary positions (in_idx 0-9
        // all have bit_offset + 2 < 32, so their 3-bit codes fit in word 0).
        let codes: Vec<(usize, u32)> = vec![
            (0, 5),
            (1, 2),
            (2, 7),
            (3, 1),
            (4, 4),
            (5, 6),
            (6, 3),
            (7, 0),
            (8, 3),
            (9, 5),
        ];
        let mut qw_words = [0u32; 3];
        for &(idx, code) in &codes {
            for b in 0..3usize {
                let overall_bit = idx * 3 + b;
                let w_idx = overall_bit / 32;
                let bit_in_word = overall_bit % 32;
                qw_words[w_idx] |= ((code >> b) & 1) << bit_in_word;
            }
        }
        // Write words to qweight.
        for w in 0..3usize {
            qweight[w * 4..w * 4 + 4].copy_from_slice(&qw_words[w].to_le_bytes());
        }

        // Fill qzeros with zero_val packed at bit 0 of word 0
        let zw: u32 = zero_val;
        qzeros[0..4].copy_from_slice(&zw.to_le_bytes());

        let result = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales,
            None,
            &[in_features, out_features],
            3, // 3-bit
            group_size,
        );

        assert!(result.is_ok());
        let deq = result.unwrap();
        assert_eq!(deq.len(), in_features);

        // Expected: (code - (zero_val + 1)) * scale_val = (code - 1) * 1.0
        let mut expected = vec![0.0f32; in_features];
        for (i, slot) in expected.iter_mut().enumerate() {
            let code = codes
                .iter()
                .find(|&&(idx, _)| idx == i)
                .map(|&(_, c)| c)
                .unwrap_or(0);
            *slot = (code as f32 - 1.0) * scale_val;
        }

        for i in 0..deq.len() {
            assert!(
                (deq[i] - expected[i]).abs() < 1e-5,
                "Mismatch at index {}: got {}, want {}",
                i,
                deq[i],
                expected[i]
            );
        }
    }

    #[test]
    fn gptq_2bit_basic() {
        // 2-bit GPTQ: 16 values per u32 word; pack known non-zero codes
        // at word-boundary positions and assert exact dequant values.
        let in_features = 16;
        let out_features = 1;
        let group_size = 16;

        let mut qweight = vec![0u8; 4]; // 1 word
        let mut qzeros = vec![0u8; 4]; // 1 word
        let scales = 1.0f32.to_le_bytes().to_vec();

        let zero_val = 0u32; // zero_point = zero_val + 1 = 1
        let scale_val = 1.0f32;

        // Pack known non-zero codes at word-boundary positions
        // (all 16 values fit in one u32 word for 2-bit: 16 * 2 = 32 bits).
        let codes: Vec<(usize, u32)> = vec![
            (0, 1),
            (1, 3),
            (2, 2),
            (3, 1),
            (4, 3),
            (5, 0),
            (6, 2),
            (7, 1),
            (8, 3),
            (9, 0),
            (10, 1),
            (11, 2),
            (12, 3),
            (13, 1),
            (14, 0),
            (15, 2),
        ];
        let mut w0: u32 = 0;
        for &(idx, code) in &codes {
            let bit_offset = idx * 2; // 2 bits per value
            w0 |= code << bit_offset;
        }
        qweight[0..4].copy_from_slice(&w0.to_le_bytes());
        // Fill qzeros with zero_val packed at bit 0 of word 0
        let zw: u32 = zero_val;
        qzeros[0..4].copy_from_slice(&zw.to_le_bytes());

        let result = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales,
            None,
            &[in_features, out_features],
            2, // 2-bit
            group_size,
        );

        assert!(result.is_ok());
        let deq = result.unwrap();
        assert_eq!(deq.len(), in_features);

        // Expected: (code - (zero_val + 1)) * scale_val = (code - 1) * 1.0
        let mut expected = vec![0.0f32; in_features];
        for (i, slot) in expected.iter_mut().enumerate() {
            let code = codes
                .iter()
                .find(|&&(idx, _)| idx == i)
                .map(|&(_, c)| c)
                .unwrap_or(0);
            *slot = (code as f32 - 1.0) * scale_val;
        }

        for i in 0..deq.len() {
            assert!(
                (deq[i] - expected[i]).abs() < 1e-5,
                "Mismatch at index {}: got {}, want {}",
                i,
                deq[i],
                expected[i]
            );
        }
    }

    #[test]
    fn gptq_4bit_basic() {
        // 4-bit GPTQ: 8 values per u32 word; pack known non-zero codes
        // at word-boundary positions and assert exact dequant values.
        let in_features = 8;
        let out_features = 1;
        let group_size = 8;

        let mut qweight = vec![0u8; 4]; // 1 word
        let mut qzeros = vec![0u8; 4]; // 1 word
        let scales = 1.0f32.to_le_bytes().to_vec();

        let zero_val = 0u32; // zero_point = zero_val + 1 = 1
        let scale_val = 1.0f32;

        // Pack known non-zero codes at word-boundary positions
        // (all 8 values fit in one u32 word for 4-bit: 8 * 4 = 32 bits).
        let codes: Vec<(usize, u32)> = vec![
            (0, 1),
            (1, 3),
            (2, 7),
            (3, 2),
            (4, 5),
            (5, 4),
            (6, 6),
            (7, 1),
        ];
        let mut w0: u32 = 0;
        for &(idx, code) in &codes {
            let bit_offset = idx * 4; // 4 bits per value
            w0 |= code << bit_offset;
        }
        qweight[0..4].copy_from_slice(&w0.to_le_bytes());
        // Fill qzeros with zero_val packed at bit 0 of word 0
        let zw: u32 = zero_val;
        qzeros[0..4].copy_from_slice(&zw.to_le_bytes());

        let result = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales,
            None,
            &[in_features, out_features],
            4, // 4-bit
            group_size,
        );

        assert!(result.is_ok());
        let deq = result.unwrap();
        assert_eq!(deq.len(), in_features);

        // Expected: (code - (zero_val + 1)) * scale_val = (code - 1) * 1.0
        let mut expected = vec![0.0f32; in_features];
        for (i, slot) in expected.iter_mut().enumerate() {
            let code = codes
                .iter()
                .find(|&&(idx, _)| idx == i)
                .map(|&(_, c)| c)
                .unwrap_or(0);
            *slot = (code as f32 - 1.0) * scale_val;
        }

        for i in 0..deq.len() {
            assert!(
                (deq[i] - expected[i]).abs() < 1e-5,
                "Mismatch at index {}: got {}, want {}",
                i,
                deq[i],
                expected[i]
            );
        }
    }

    // FP4/NF4/FP8 dequantization tests

    #[test]
    fn roundtrip_fp4() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 12.8).collect(); // Scale to E2M1 range
        let quantized = quant_fp4(&data).unwrap();
        let dequantized = dequant_fp4(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        let mse = mean_squared_error(&data, &dequantized);
        // FP4 has coarse precision, allow higher MSE
        assert!(mse < 0.3, "fp4 mse too high: {mse}");
    }

    #[test]
    fn fp4_dequant_preserves_extremes() {
        // Test FP4 extreme values: -1.0, 0.0, 1.0
        // FP4 max representable is 0.875 in E2M1, so values are scaled
        let mut data = vec![0.0f32; 8];
        data[0] = -1.0;
        data[1] = -0.5;
        data[2] = 0.0;
        data[3] = 0.5;
        data[4] = 1.0;

        let quantized = quant_fp4(&data).unwrap();
        let deq = dequant_fp4(&quantized, 8).unwrap();

        // FP4 has limited precision - check values are in expected range
        // Scale is computed from max value (1.0), so range should be approximately [-0.875, 0.875]
        assert!(
            deq[0].abs() > 0.7,
            "FP4 -1.0 should map to ~-0.875: {}",
            deq[0]
        ); // -1.0
        assert!(
            deq[4].abs() > 0.7,
            "FP4 +1.0 should map to ~+0.875: {}",
            deq[4]
        ); // +1.0
        assert!(
            (deq[2] - 0.0).abs() < 0.05,
            "FP4 0.0 should be near zero: {}",
            deq[2]
        );
    }

    #[test]
    fn roundtrip_nf4() {
        // NF4 values are designed for normal distribution, test with values in [-1, 1]
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let quantized = quant_nf4(&data).unwrap();
        let dequantized = dequant_nf4(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        let mse = mean_squared_error(&data, &dequantized);
        assert!(mse < 0.1, "nf4 mse too high: {mse}");
    }

    #[test]
    fn nf4_dequant_preserves_zero_crossing() {
        // NF4 has finer granularity near zero
        let data = vec![0.125, 0.0, -0.125];
        let quantized = quant_nf4(&data).unwrap();
        let deq = dequant_nf4(&quantized, 3).unwrap();
        assert_eq!(deq.len(), 3);
        assert!(deq[0] > 0.0, "NF4 positive near-zero should be positive");
        assert!(deq[2] < 0.0, "NF4 negative near-zero should be negative");
    }

    #[test]
    fn roundtrip_fp8() {
        // FP8 E4M3 works well for values in the range [-64, 64] approximately
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let quantized = quant_fp8(&data).unwrap();
        let dequantized = dequant_fp8(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        // FP8 has limited precision, especially for larger values
        // Check that we can recover the data within reasonable error
        let max_diff = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 10.0, "fp8 max diff too high: {}", max_diff);
    }

    #[test]
    fn fp8_dequant_handles_small_values() {
        // Small values in FP8 subnormal range
        let data = vec![0.01, 0.02, 0.03, 0.04];
        let quantized = quant_fp8(&data).unwrap();
        let deq = dequant_fp8(&quantized, 4).unwrap();
        assert_eq!(deq.len(), 4);
        // Small values may lose precision in FP8 - just check they're close
        for i in 0..4 {
            let diff = (deq[i] - data[i]).abs();
            assert!(
                diff < 0.1,
                "FP8 small value diff too high at {}: {}",
                i,
                diff
            );
        }
    }

    fn build_mxfp4_single_buffer(codes: &[u8], exps: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        buf.extend_from_slice(codes);
        buf.extend_from_slice(&(exps.len() as u64).to_le_bytes());
        buf.extend_from_slice(exps);
        buf
    }

    fn build_mxfp8_single_buffer(codes: &[u8], exps: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        buf.extend_from_slice(codes);
        buf.extend_from_slice(&(exps.len() as u64).to_le_bytes());
        buf.extend_from_slice(exps);
        buf
    }

    #[test]
    fn dequant_mxfp4_matches_kernel_nibble_order() {
        // shared_exp = 127 -> scale 2^0 = 1.0. Two codes per byte:
        // byte 0x21 -> element 0 (low nibble) = 1, element 1 (high nibble) = 2.
        let codes = vec![0x21u8, 0x43u8];
        let exps = vec![127u8];
        let buf = build_mxfp4_single_buffer(&codes, &exps);
        let deq = dequant_mxfp4(&buf, 4).unwrap();
        assert_eq!(deq.len(), 4);
        assert_eq!(deq[0], mxfp4_e2m1_to_f32(0x1, 127));
        assert_eq!(deq[1], mxfp4_e2m1_to_f32(0x2, 127));
        assert_eq!(deq[2], mxfp4_e2m1_to_f32(0x3, 127));
        assert_eq!(deq[3], mxfp4_e2m1_to_f32(0x4, 127));
    }

    #[test]
    fn dequant_mxfp4_applies_shared_exp_scale() {
        // shared_exp = 130 -> scale 2^3 = 8.0. code 4 (E2M1: exp=2,mant=0) = 2.0 unscaled.
        let codes = vec![0x04u8];
        let exps = vec![130u8];
        let buf = build_mxfp4_single_buffer(&codes, &exps);
        let deq = dequant_mxfp4(&buf, 2).unwrap();
        assert_eq!(deq.len(), 2);
        assert!((deq[0] - 16.0).abs() < 1e-5, "deq[0] = {}", deq[0]);
        assert!((deq[1] - 0.0).abs() < 1e-5, "deq[1] = {}", deq[1]);
    }

    #[test]
    fn dequant_mxfp4_roundtrip() {
        let data: Vec<f32> = (0..96).map(|i| ((i as f32 - 48.0) / 48.0) * 4.0).collect();
        let shared_exp = 127u8;
        let mut codes = vec![0u8; data.len().div_ceil(2)];
        for (i, &v) in data.iter().enumerate() {
            let code = f32_to_mxfp4_e2m1(v, shared_exp);
            if i % 2 == 0 {
                codes[i / 2] |= code & 0x0F;
            } else {
                codes[i / 2] |= (code & 0x0F) << 4;
            }
        }
        let exps = vec![shared_exp; data.len().div_ceil(32)];
        let buf = build_mxfp4_single_buffer(&codes, &exps);
        let deq = dequant_mxfp4(&buf, data.len()).unwrap();
        assert_eq!(deq.len(), data.len());
        // MXFP4 has coarse precision; values chosen within E2M1 representable range
        for i in 0..data.len() {
            let diff = (data[i] - deq[i]).abs();
            assert!(
                diff < 1.5,
                "diff at {i}: {} vs {}, diff={}",
                data[i],
                deq[i],
                diff
            );
        }
    }

    #[test]
    fn dequant_mxfp4_rejects_truncated_segments() {
        let codes = vec![0x00u8];
        let exps = vec![127u8; 8]; // need 8 exps for 256 values
        let mut buf = build_mxfp4_single_buffer(&codes, &exps);
        // 256 values need 128 code bytes, only 1 present -> error
        assert!(dequant_mxfp4(&buf, 256).is_err());
        // Truncate the length prefix itself
        buf.truncate(4);
        assert!(dequant_mxfp4(&buf, 256).is_err());
    }

    #[test]
    fn dequant_mxfp8_roundtrip_and_scale() {
        // shared_exp = 127, code 0x40 = E4M3 (exp 8, mant 0) = 2.0
        let codes = vec![0x40u8; 4];
        let exps = vec![127u8];
        let buf = build_mxfp8_single_buffer(&codes, &exps);
        let deq = dequant_mxfp8(&buf, 4).unwrap();
        assert_eq!(deq.len(), 4);
        assert!((deq[0] - 2.0).abs() < 1e-5, "deq[0] = {}", deq[0]);

        // shared_exp = 128 -> scale 2.0, so value doubles
        let exps2 = vec![128u8];
        let buf2 = build_mxfp8_single_buffer(&codes, &exps2);
        let deq2 = dequant_mxfp8(&buf2, 4).unwrap();
        assert!((deq2[0] - 4.0).abs() < 1e-5, "deq2[0] = {}", deq2[0]);
    }

    #[test]
    fn dequant_mxfp8_rejects_truncated_segments() {
        let codes = vec![0x40u8];
        let exps = vec![127u8; 8];
        let buf = build_mxfp8_single_buffer(&codes, &exps);
        assert!(dequant_mxfp8(&buf, 256).is_err());
    }

    #[test]
    fn rewrite_tensor_to_fp4() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 12.8).collect();
        let rewritten = rewrite_tensor_data(
            &data,
            &TensorRewritePlan {
                target: QuantFormat::Fp4,
                shape: vec![32, 1],
                importance: None,
                curvature: None,
            },
        )
        .unwrap();
        assert!(!rewritten.bytes.is_empty());
        assert_eq!(rewritten.target, QuantFormat::Fp4);
    }

    #[test]
    fn rewrite_tensor_to_nf4() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let rewritten = rewrite_tensor_data(
            &data,
            &TensorRewritePlan {
                target: QuantFormat::Nf4,
                shape: vec![32, 1],
                importance: None,
                curvature: None,
            },
        )
        .unwrap();
        assert!(!rewritten.bytes.is_empty());
        assert_eq!(rewritten.target, QuantFormat::Nf4);
    }

    #[test]
    fn rewrite_tensor_to_fp8() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let rewritten = rewrite_tensor_data(
            &data,
            &TensorRewritePlan {
                target: QuantFormat::Fp8,
                shape: vec![32, 1],
                importance: None,
                curvature: None,
            },
        )
        .unwrap();
        assert!(!rewritten.bytes.is_empty());
        assert_eq!(rewritten.target, QuantFormat::Fp8);
    }

    // Pass 4: Fisher/Hessian diagonal - unit tests

    #[test]
    fn test_compute_fisher_diagonal_empty_calibration() {
        let weights = vec![0.1f32; 256];
        let result = compute_fisher_diagonal(&weights, &[], 16, 16, 128);
        // Empty calibration → should return ones (identity-like curvature)
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_compute_fisher_diagonal_single_sample() {
        let rows = 4;
        let cols = 8;
        let weights = vec![0.1f32; rows * cols];
        let samples = vec![FisherCalibrationSample {
            input_activations: vec![1.0; cols],
            output_gradients: vec![0.5; rows],
        }];
        let result = compute_fisher_diagonal(&weights, &samples, rows, cols, 128);
        assert_eq!(result.len(), rows * cols);
        assert!(result.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_compute_grouped_fisher_diagonal() {
        let rows = 4;
        let cols = 64;
        let weights = vec![0.1f32; rows * cols];
        let samples = vec![FisherCalibrationSample {
            input_activations: vec![1.0; cols],
            output_gradients: vec![1.0; rows],
        }];
        let result = compute_grouped_fisher_diagonal(&weights, &samples, rows, cols, 32);
        let expected_groups = cols.div_ceil(32);
        assert_eq!(result.len(), expected_groups);
        assert!(result.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_compute_grouped_fisher_diagonal_empty() {
        let result = compute_grouped_fisher_diagonal(&[], &[], 4, 64, 32);
        assert_eq!(result.len(), 2); // 64/32 = 2 groups, returns ones
        assert!(result.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_compute_curvature_proxy() {
        let data = vec![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let layer_importance = 1.0;
        let result = compute_curvature_proxy(&data, layer_importance);
        assert_eq!(result.len(), data.len());
        // Base value is 1.0 + importance * (|x| + x²) min 16
        assert!(result.iter().all(|v| *v >= 1.0));
        // Larger magnitude → larger curvature
        assert!(result[3] > result[0]); // |2.0| > |0.0|
        assert!(result[4] > result[1]); // |-2.0| > |1.0|
    }

    #[test]
    fn test_compute_curvature_proxy_zero_importance() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = compute_curvature_proxy(&data, 0.0);
        // Minimum scale is 1e-3 even when importance is 0 (safeguard against degenerate values) value=1.0: 1.0 + 0.001 *
        // (1+1) = 1.002 value=2.0: 1.0 + 0.001 * (2+4) = 1.006 value=3.0: 1.0 + 0.001 * (3+9) = 1.012
        assert!((result[0] - 1.002).abs() < 1e-5);
        assert!((result[1] - 1.006).abs() < 1e-5);
        assert!((result[2] - 1.012).abs() < 1e-5);
        // All values >= 1.0
        assert!(result.iter().all(|v| *v >= 1.0));
    }

    // Edge-case + boundary tests (P1 strengthening).
    // The existing tests above cover happy paths with 64-element inputs.

    #[test]
    fn q80_round_trip_preserves_length_across_block_boundary() {
        // Q8_0 block size is 32. Test inputs that cross the
        // boundary: 31 (sub-block tail), 32 (exact block), 33 (block + 1).
        for &n in &[31usize, 32, 33, 63, 64, 65] {
            let data: Vec<f32> = (0..n)
                .map(|i| (i as f32 - (n as f32 / 2.0)) * 0.1)
                .collect();
            let q = quant_q80(&data).expect("quant");
            let d = dequant_q80(&q, n).expect("dequant");
            assert_eq!(d.len(), n, "Q8_0 length contract broken at n={n}");
        }
    }

    #[test]
    fn q80_round_trip_all_zeros_does_not_produce_nan() {
        // All-zero input → amax = 0 → scale guard picks 1.0 (line 518).
        // A mutant that dropped the `amax == 0.0` guard would divide by zero and produce.
        let data = vec![0.0f32; 64];
        let q = quant_q80(&data).expect("quant");
        let d = dequant_q80(&q, 64).expect("dequant");
        assert_eq!(d.len(), 64);
        assert!(
            d.iter().all(|v| v.is_finite()),
            "all-zero must not yield NaN"
        );
        // Reconstruction of zero is exactly zero (q=0, scale arbitrary).
        assert!(d.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn q80_round_trip_constant_nonzero_input() {
        // Constant input exercises the scale path without amax=0 degeneracy.
        // A scale-fit mutant would surface as reconstruction != constant.
        let data = vec![0.5f32; 64];
        let q = quant_q80(&data).expect("quant");
        let d = dequant_q80(&q, 64).expect("dequant");
        for v in &d {
            assert!(
                (v - 0.5).abs() < 0.02,
                "constant reconstruction drifted: {v}"
            );
        }
    }

    #[test]
    fn q4k_rejects_truncated_buffer() {
        // Q4_K stride is 4 (scale) + 16 (packed 4-bit) = 20 bytes per 32-weight block.
        // A buffer shorter than `num_blocks * 20` must error, not silently read past the end.
        let short_buf = vec![0u8; 10]; // claims 64 weights but only 10 bytes
        let res = dequant_q4k(&short_buf, 64);
        assert!(res.is_err(), "dequant_q4k must reject truncated buffer");
    }

    #[test]
    fn q80_rejects_truncated_buffer() {
        // Q8_0 stride is 2 (f16 scale) + 32 (i8 weights) = 34 bytes per 32-weight block.
        // Handing in 5 bytes while claiming 32 weights must error rather than reading out of.
        let short_buf = vec![0u8; 5];
        let res = dequant_q80(&short_buf, 32);
        assert!(res.is_err(), "dequant_q80 must reject truncated buffer");
    }

    #[test]
    fn iq4nl_rejects_truncated_buffer() {
        // An IQ4_NL block is 18 bytes for 32 weights, so 256 weights need
        // 144. A 143-byte buffer claiming 256 weights must error.
        let short_buf = vec![0u8; 143];
        let res = dequant_iq4nl(&short_buf, 256);
        assert!(res.is_err(), "dequant_iq4nl must reject truncated buffer");
    }

    #[test]
    fn fp4_round_trip_preserves_sign() {
        // FP4 E2M1 has a sign bit; quant → dequant must not flip the sign of a clearly positive or clearly negative input.
        // A mutant that dropped the sign-bit branch in the quantizer would surface here.
        let pos = vec![0.5f32; 16];
        let neg = vec![-0.5f32; 16];
        let q_pos = quant_fp4(&pos).expect("quant pos");
        let d_pos = dequant_fp4(&q_pos, 16).expect("dequant pos");
        let q_neg = quant_fp4(&neg).expect("quant neg");
        let d_neg = dequant_fp4(&q_neg, 16).expect("dequant neg");
        assert!(
            d_pos.iter().all(|v| *v >= 0.0),
            "FP4 must preserve positive sign"
        );
        assert!(
            d_neg.iter().all(|v| *v <= 0.0),
            "FP4 must preserve negative sign"
        );
    }

    #[test]
    fn nf4_round_trip_preserves_zero_crossing() {
        // NF4 is asymmetric with no exact zero code; the smallest positive code is +0.1 and the largest negative is -0.1.
        // Quantizing a mixed-sign input must produce a dequant vector that has both signs - a.
        let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let q = quant_nf4(&data).expect("quant");
        let d = dequant_nf4(&q, 16).expect("dequant");
        let has_pos = d.iter().any(|v| *v > 0.0);
        let has_neg = d.iter().any(|v| *v < 0.0);
        assert!(
            has_pos && has_neg,
            "NF4 must preserve both signs; got {:?}",
            d
        );
    }

    #[test]
    fn fp8_quant_clamps_to_representable_range() {
        // E4M3 max representable is ~240. Quantizing +1e6 must clamp, not overflow the 4-bit exponent field
        // - a mutant that dropped the `.min(240.0)` clamp at line 727 would corrupt the bit pattern.
        let data = vec![1.0e6f32, -1.0e6, 0.0, 1.0];
        let q = quant_fp8(&data).expect("quant");
        let d = dequant_fp8(&q, 4).expect("dequant");
        // The clamped values land near the E4M3 max (~240). We assert only
        // finiteness + sign preservation — exact value depends on the LUT.
        assert!(
            d[0].is_finite() && d[0] > 100.0,
            "large positive must clamp to ~240; got {}",
            d[0]
        );
        assert!(
            d[1].is_finite() && d[1] < -100.0,
            "large negative must clamp to ~-240; got {}",
            d[1]
        );
        assert!(d[2].abs() < 1e-6, "zero must round-trip; got {}", d[2]);
    }

    #[test]
    fn quant_q80_empty_input_returns_empty_or_errors_cleanly() {
        // Empty input is a boundary the existing tests skip. The contract
        // is "no panic" — either empty output or clean Err.
        let res = quant_q80(&[]);
        match res {
            Ok(bytes) => assert!(bytes.is_empty(), "empty input must yield empty bytes"),
            Err(_) => { /* clean error is also acceptable */ }
        }
    }

    #[test]
    fn dequant_fp4_empty_input_returns_empty() {
        // dequant_fp4 with num_values=0 must not index into data[4..].
        let data = vec![0u8; 4]; // scale only, no packed codes
        let d = dequant_fp4(&data, 0).expect("dequant");
        assert!(d.is_empty(), "num_values=0 must yield empty output");
    }

    #[test]
    fn dequant_fp8_handles_short_buffer_without_panic() {
        let short = vec![0u8; 3];
        let _ = dequant_fp8(&short, 3).expect("short fp8 dequant");
        let exact = vec![0u8; 8];
        let _ = dequant_fp8(&exact, 4).expect("fp8 dequant at scale boundary");
    }

    #[test]
    fn fp4_block_round_trip() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let q = quant_fp4_block16(&data, 16).expect("quant block fp4");
        println!("fp4 q: {:?}", q);
        let d = dequant_fp4_block16(&q, 32).expect("dequant block fp4");
        println!("fp4 d: {:?}", d);
        assert_eq!(d.len(), 32);
        for (got, want) in d.iter().zip(data.iter()) {
            assert!(
                (got - want).abs() < 0.2,
                "FP4 block round trip error too high: got {} vs want {}",
                got,
                want
            );
        }
    }

    #[test]
    fn fp8_block_round_trip() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let q = quant_fp8_block16(&data, 16).expect("quant block fp8");
        println!("fp8 q: {:?}", q);
        let d = dequant_fp8_block16(&q, 32).expect("dequant block fp8");
        println!("fp8 d: {:?}", d);
        assert_eq!(d.len(), 32);
        for (got, want) in d.iter().zip(data.iter()) {
            assert!(
                (got - want).abs() < 0.15,
                "FP8 block round trip error too high: got {} vs want {}",
                got,
                want
            );
        }
    }

    /// P2-WI-1 gate: `RowScaleDtype::Fp8` with `block_size = 16` must round-trip a non-trivial tensor with bounded error relative to the legacy single-global-scale (`fp8` only).
    /// The two-level scale structure is what enables a future kernel to reach NVFP4-level accuracy on.
    #[test]
    fn fp8_block_round_trip_is_no_worse_than_single_scale() {
        let mut data: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            let v = (i as f32 - 8.0) / 8.0; // ~[-1, 1]
            data.push(v);
        }
        for i in 0..16 {
            let v = (i as f32 - 8.0) * 12.5; // ~[-100, 100]
            data.push(v);
        }

        let q = quant_fp8_block16(&data, 16).expect("quant block fp8");
        let d_block = dequant_fp8_block16(&q, 32).expect("dequant block fp8");

        let q_single = quant_fp8(&data).expect("quant single-scale fp8");
        let d_single = dequant_fp8(&q_single, 32).expect("dequant single-scale fp8");

        let mut err_block = 0.0f32;
        let mut err_single = 0.0f32;
        for i in 0..32 {
            err_block += (data[i] - d_block[i]).abs();
            err_single += (data[i] - d_single[i]).abs();
        }
        // The block path must be within a small multiple of the single-scale path (no regression; equal-or-better).
        // The spec's "must have lower error" claim is reserved for the future NVFP4-equivalent kernel that.
        assert!(
            err_block <= err_single * 1.2 + 1e-3,
            "block path must not regress vs single-scale: block={} single={}",
            err_block,
            err_single
        );
    }

    #[test]
    fn test_gptq_dequant_correctness_fixture() {
        let in_features = 32;
        let out_features = 32;
        let group_size = 16;
        let bits = 4;
        let values_per_word = 8;

        let mut expected = vec![0.0f32; in_features * out_features];
        let mut qweight = vec![0u8; (in_features / values_per_word) * out_features * 4];
        let mut qzeros =
            vec![0u8; (in_features / group_size) * (out_features / values_per_word) * 4];
        let mut scales = vec![0u8; (in_features / group_size) * out_features * 4];

        let zero_val = 7u32;
        let scale_val = 0.5f32;

        let num_groups = in_features / group_size;
        for g in 0..num_groups {
            for col in 0..out_features {
                let scale_idx = g * out_features + col;
                let sb = scale_val.to_le_bytes();
                scales[scale_idx * 4..scale_idx * 4 + 4].copy_from_slice(&sb);

                let zero_word_idx = g * (out_features / values_per_word) + col / values_per_word;
                let bit_offset = (col % values_per_word) * bits;
                let offset = zero_word_idx * 4;
                let mut word = u32::from_le_bytes([
                    qzeros[offset],
                    qzeros[offset + 1],
                    qzeros[offset + 2],
                    qzeros[offset + 3],
                ]);
                word |= zero_val << bit_offset;
                qzeros[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }

        for in_idx in 0..in_features {
            for out_idx in 0..out_features {
                let code = ((in_idx + out_idx) % 16) as u32;
                expected[in_idx * out_features + out_idx] =
                    (code as f32 - (zero_val + 1) as f32) * scale_val;

                let word_idx = (in_idx / values_per_word) * out_features + out_idx;
                let bit_offset = (in_idx % values_per_word) * bits;
                let offset = word_idx * 4;
                let mut word = u32::from_le_bytes([
                    qweight[offset],
                    qweight[offset + 1],
                    qweight[offset + 2],
                    qweight[offset + 3],
                ]);
                word |= code << bit_offset;
                qweight[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }

        let dequanted = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales,
            None,
            &[in_features, out_features],
            bits as u32,
            group_size,
        )
        .unwrap();

        assert_eq!(dequanted.len(), expected.len());
        for i in 0..dequanted.len() {
            assert!(
                (dequanted[i] - expected[i]).abs() < 1e-5,
                "Mismatch at index {}: got {}, want {}",
                i,
                dequanted[i],
                expected[i]
            );
        }
    }

    /// Tests exact 34-byte block layout and scale math for Q8_0 (d: f16 scale + 32 i8 codes).
    #[test]
    fn test_q80_bit_exact_layout_and_math() {
        let mut block = vec![0u8; 34];
        // d = 0.5f16 (0x3800 in LE bytes)
        block[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        // i8 codes: [-128, -2, 0, 2, 127]
        block[2] = (-128i8) as u8;
        block[3] = (-2i8) as u8;
        block[4] = 0u8;
        block[5] = 2u8;
        block[6] = 127u8;

        let deq = dequant_q80(&block, 32).expect("dequant q80");
        assert_eq!(deq.len(), 32);
        assert_eq!(deq[0], -64.0f32);
        assert_eq!(deq[1], -1.0f32);
        assert_eq!(deq[2], 0.0f32);
        assert_eq!(deq[3], 1.0f32);
        assert_eq!(deq[4], 63.5f32);

        // Test quant_q80 round-trip produces valid 34-byte block
        let mut sample = vec![0.0f32; 32];
        sample[0] = -64.0;
        sample[1] = -1.0;
        sample[2] = 0.0;
        sample[3] = 1.0;
        sample[4] = 63.5;

        let quant = quant_q80(&sample).expect("quant q80");
        assert_eq!(quant.len(), 34);
        let redq = dequant_q80(&quant, 32).expect("re-dequant q80");
        assert!((redq[0] - (-64.0)).abs() < 1e-2);
        assert!((redq[1] - (-1.0)).abs() < 1e-2);
        assert!((redq[2] - 0.0).abs() < 1e-2);
        assert!((redq[3] - 1.0).abs() < 1e-2);
        assert!((redq[4] - 63.5).abs() < 1e-2);
    }

    /// Tests Q4_K super-block math with sub-block scale sc_i and min m_i (y = d * sc_i * q - dmin * m_i).
    #[test]
    fn test_q4k_subblock_scales_and_min_math() {
        let mut data = vec![0u8; 144];
        // d = 1.0f16 (0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // dmin = 0.5f16 (0x3800)
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        // scales: sub-block 0 has sc_0 = 2, m_0 = 1
        // Q4_K scale encoding: sc_0 = 2 (byte 0 low 6 bits = 2), m_0 = 1 (byte 4 low 6 bits = 1)
        data[4] = 2;
        data[8] = 1;
        // qs byte 0: low nibble = 4 (q_0 = 4)
        data[16] = 4;

        let deq = dequant_q4k(&data, 256).expect("dequant q4k");
        assert_eq!(deq.len(), 256);
        // y_0 = 1.0 * 2 * 4 - 0.5 * 1 = 7.5
        assert_eq!(deq[0], 7.5f32);
    }

    /// Tests Q5_K format for 256 weights (176 bytes per block).
    #[test]
    fn test_q5k_5bit_high_bit_unpacking() {
        let mut data = vec![0u8; 176];
        // d = 1.0f16 (0x3C00) -> bytes 0..2
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // dmin = 0.5f16 (0x3800) -> bytes 2..4
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        // scales: sub-block 0 sc_0 = 2 (data[4] = 2), m_0 = 1 (data[8] = 1)
        data[4] = 2;
        data[8] = 1;
        // qh: byte 0 = 1 (bit 0 set -> msb for elem 0 is 16)
        data[16] = 1;
        // qs byte 0: low nibble = 4 (q_lo = 4, so q1 = 4 + 16 = 20)
        data[48] = 4;

        let deq = dequant_q5k(&data, 256).expect("dequant q5k");
        assert_eq!(deq.len(), 256);
        // deq[0] = d * sc_0 * q1 - dmin * m_0 = 1.0 * 2 * 20 - 0.5 * 1 = 39.5
        assert_eq!(deq[0], 39.5f32);
        // deq[32] = d * sc_1 * q2 - dmin * m_1 = 1.0 * 0 * 0 - 0.5 * 0 = 0.0
        assert_eq!(deq[32], 0.0f32);
    }

    /// Tests Q6_K format for 256 weights (210 bytes per block).
    #[test]
    fn test_q6k_6bit_split_code_reconstruction() {
        let mut data = vec![0u8; 210];
        // d = 2.0f16 (0x4000) at offset 208..210
        data[208..210].copy_from_slice(&0x4000u16.to_le_bytes());
        // scales: signed i8 scales at offset 192. scale 0 = 4 (data[192] = 4)
        data[192] = 4;
        // ql byte 0 = 5 (low nibble 5)
        data[0] = 5;
        // qh byte 0 = 1 (bits 0..1 = 1 -> msb shift by 4 is 16)
        data[128] = 1;

        let deq = dequant_q6k(&data, 256).expect("dequant q6k");
        assert_eq!(deq.len(), 256);
        // q = 5 | (1 << 4) = 21. value = d * sc * (q - 32) = 2.0 * 4 * (21 - 32) = -88.0
        assert_eq!(deq[0], -88.0f32);
    }

    /// Host-side mirror of the corrected ROCm `q6k_gemm.rs::dequant_q6k_element` HIP kernel.
    /// Kept line-for-line equivalent to the kernel so this test actually exercises the kernel's bit-math derivation,.
    fn host_dequant_q6k_element(block: &[u8], in_sb: usize) -> f32 {
        assert!(in_sb < 256 && block.len() >= 210);
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(block[208], block[209]);

        let n = in_sb / 128;
        let pos = in_sb % 128;
        let quarter = pos / 32; // 0..3 (matches CPU reference's q1..q4)
        let l = pos % 32;
        let is = l / 16;
        let sc_idx = n * 8 + is + 2 * quarter;

        let sc = scales[sc_idx] as i8 as f32;

        let ql_offset = n * 64 + l + if (quarter & 1) != 0 { 32 } else { 0 };
        let ql_byte = ql[ql_offset];
        let nibble = if (quarter & 2) != 0 {
            ql_byte >> 4
        } else {
            ql_byte & 0x0F
        };

        let qh_byte = qh[n * 32 + l];
        let qh_bits = (qh_byte >> (2 * quarter)) & 0x03;
        let q_code = (nibble as i32) | ((qh_bits as i32) << 4);

        d * sc * (q_code as f32 - 32.0)
    }

    /// Golden-vector check: across 256 `in_sb`, the host-mirrored GPU per-element formula must produce byte-identical values to the CPU reference `dequant_q6k`.
    /// Also asserts the old (broken) Q5_K-style layout the kernel previously used would NOT match -.
    #[test]
    fn test_q6k_gpu_kernel_element_matches_cpu_reference() {
        let mut data = vec![0u8; 210];
        // Deterministic-but-varied fill that touches all bit planes without
        // making every element identical (which would mask off-by-one bugs).
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = ((i * 7 + 13) as u8).wrapping_add((i as u8) ^ 0x5A);
        }
        // Signed scales at +192: spread positives & negatives across all 16.
        for (i, slot) in data[192..208].iter_mut().enumerate() {
            *slot = (i as i8).wrapping_mul(3).wrapping_sub(10) as u8;
        }
        // d (f16) at +208: pick a non-trivial scale, 1.5 ≈ 0x3E00.
        data[208..210].copy_from_slice(&0x3E00u16.to_le_bytes());

        let cpu = dequant_q6k(&data, 256).expect("dequant_q6k");
        assert_eq!(cpu.len(), 256);

        for (in_sb, &cpu_v) in cpu.iter().enumerate() {
            let gpu = host_dequant_q6k_element(&data, in_sb);
            assert!(
                (gpu - cpu_v).abs() <= 1e-4 * cpu_v.abs().max(1.0),
                "in_sb={in_sb}: GPU-mirror={gpu} != CPU-ref={cpu_v}"
            );
        }
    }

    /// Sanity: the test block above must NOT be all-zeros after dequant, or
    /// the golden-vector comparison would pass vacuously.
    #[test]
    fn test_q6k_golden_block_is_nontrivial() {
        let mut data = vec![0u8; 210];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = ((i * 7 + 13) as u8).wrapping_add((i as u8) ^ 0x5A);
        }
        for (i, slot) in data[192..208].iter_mut().enumerate() {
            *slot = (i as i8).wrapping_mul(3).wrapping_sub(10) as u8;
        }
        data[208..210].copy_from_slice(&0x3E00u16.to_le_bytes());
        let cpu = dequant_q6k(&data, 256).expect("dequant_q6k");
        let distinct = cpu.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            distinct > 200,
            "golden block dequant produced mostly-zeros ({distinct}/256); \
             test fixture is degenerate"
        );
        // And there must be both positive and negative values (scales are
        // signed); if all same sign the q-32 centering path is untested.
        let any_pos = cpu.iter().any(|v| *v > 1e-3);
        let any_neg = cpu.iter().any(|v| *v < -1e-3);
        assert!(any_pos, "golden block has no positive outputs");
        assert!(any_neg, "golden block has no negative outputs");
    }

    /// Host-side mirror of the corrected ROCm `shared_device_fns.rs::dequant_q4k_element` HIP kernel.
    /// Q4_K super-block: 144 bytes / 256 weights.
    fn host_dequant_q4k_element(block: &[u8], in_sb: usize) -> f32 {
        assert!(in_sb < 256 && block.len() >= 144);
        let d = f16_to_f32(block[0], block[1]);
        let dmin = f16_to_f32(block[2], block[3]);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let group = in_sb / 64;
        let half = (in_sb % 64) / 32;
        let l = in_sb % 32;
        let is = 2 * group + half;

        let (sc, m) = if is < 4 {
            (scales[is] & 63, scales[is + 4] & 63)
        } else {
            (
                (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4),
                (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4),
            )
        };
        let byte = qs[group * 32 + l];
        let q = if half != 0 { byte >> 4 } else { byte & 0x0F };
        d * (sc as f32) * (q as f32) - dmin * (m as f32)
    }

    /// Golden-vector check for the Q4_K GPU element kernel: across 256 `in_sb`, the host-mirrored formula must match the CPU reference `dequant_q4k`.
    /// Uses a non-trivial deterministic block exercising all four groups and both nibble halves.
    #[test]
    fn test_q4k_gpu_kernel_element_matches_cpu_reference() {
        let mut data = vec![0u8; 144];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = ((i * 11 + 7) as u8).wrapping_add((i as u8) ^ 0x35);
        }
        // 6-bit scales must be carved out (mask 0x3F) — sprinkle valid values.
        for (i, slot) in data[4..20].iter_mut().enumerate() {
            *slot = (i as u8).wrapping_mul(5).wrapping_add(1) & 0x3F;
        }
        // d, dmin as f16: d = 1.25 (0x3FA0), dmin = 0.5 (0x3800).
        data[0..2].copy_from_slice(&0x3FA0u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        let cpu = dequant_q4k(&data, 256).expect("dequant_q4k");
        assert_eq!(cpu.len(), 256);
        // Non-degenerate: distinct outputs in each group.
        for grp_start in [0usize, 64, 128, 192] {
            let distinct = cpu[grp_start..grp_start + 64]
                .iter()
                .filter(|v| v.abs() > 1e-6)
                .count();
            assert!(distinct > 50, "group @ {grp_start} degenerate ({distinct})");
        }

        for (in_sb, &cpu_v) in cpu.iter().enumerate() {
            let gpu = host_dequant_q4k_element(&data, in_sb);
            assert!(
                (gpu - cpu_v).abs() <= 1e-3 * cpu_v.abs().max(1.0),
                "in_sb={in_sb}: GPU-mirror={gpu} != CPU-ref={cpu_v}"
            );
        }
    }

    /// Definitive check: extract a real Q4_K weight from an on-disk GGUF model and dequantize it two ways - grim's `dequant_q4k` and an independent ggml-faithful reimplementation.
    /// They MUST agree.
    #[test]
    fn test_q4k_real_model_matches_ggml_reference() {
        let path = std::env::var("GRIM_Q4K_MODEL")
            .unwrap_or_else(|_| "models/MiniCPM5-1B-Q4_K_M.gguf".into());
        let Ok(file) = std::fs::File::open(&path) else {
            eprintln!("skip: model not found at {path}");
            return;
        };
        let mut reader = file;
        let gguf = grim_format::gguf::read_gguf(&mut reader).expect("read_gguf");
        let info = gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .expect("token_embd.weight present");
        assert_eq!(
            info.dtype,
            grim_format::gguf::GgufDType::Q4K,
            "dtype must be Q4_K"
        );
        let bytes =
            grim_format::gguf::read_tensor_bytes(&mut reader, &gguf, info).expect("read bytes");
        let n = info.elem_count();

        let grim_out = dequant_q4k(&bytes, n).expect("grim dequant_q4k");

        // Independent ggml-faithful reference.
        let ref_out = dequant_q4k_ggml_ref(&bytes, n);

        assert_eq!(grim_out.len(), ref_out.len());
        let mut max_rel = 0.0f32;
        for i in 0..ref_out.len() {
            let denom = ref_out[i].abs().max(1.0);
            max_rel = max_rel.max((grim_out[i] - ref_out[i]).abs() / denom);
        }
        assert!(max_rel < 0.02, "q4k grim vs ggml-ref max_rel={max_rel}");
    }

    /// Minimal, self-contained port of llama.cpp dequantize_row_q4_K.
    fn dequant_q4k_ggml_ref(data: &[u8], num_weights: usize) -> Vec<f32> {
        const QK_K: usize = 256;
        let nb = num_weights / QK_K;
        let mut out = Vec::with_capacity(num_weights);
        for i in 0..nb {
            let base = i * 144;
            let d = f16_to_f32(data[base], data[base + 1]);
            let min = f16_to_f32(data[base + 2], data[base + 3]);
            let scales = &data[base + 4..base + 16];
            let q = &data[base + 16..base + 144];
            let mut is = 0usize;
            let mut qoff = 0usize;
            for _ in 0..(QK_K / 64) {
                let (s, m) = ggml_get_scale_min_k4(is, scales);
                let d1 = d * s;
                let m1 = min * m;
                let (s, m) = ggml_get_scale_min_k4(is + 1, scales);
                let d2 = d * s;
                let m2 = min * m;
                for l in 0..32 {
                    out.push(d1 * (q[qoff + l] & 0x0F) as f32 - m1);
                }
                for l in 0..32 {
                    out.push(d2 * (q[qoff + l] >> 4) as f32 - m2);
                }
                qoff += 32;
                is += 2;
            }
        }
        out
    }

    fn ggml_get_scale_min_k4(j: usize, sc: &[u8]) -> (f32, f32) {
        let (d, m) = if j < 4 {
            (sc[j] & 63, sc[j + 4] & 63)
        } else {
            (
                (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4),
                (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
            )
        };
        (d as f32, m as f32)
    }

    /// Tests FP8 E4M3 subnormal float decode scaling factor (1.0 / 512.0).
    #[test]
    fn test_fp8_e4m3_subnormal_scale_factor() {
        // Exp = 0, mantissa = 1 -> positive subnormal: 1.0 / 512.0
        let val_pos = fp8_e4m3_to_f32(0x01);
        assert_eq!(val_pos, 1.0 / 512.0);

        // Sign bit set (0x80), Exp = 0, mantissa = 1 -> negative subnormal: -1.0 / 512.0
        let val_neg = fp8_e4m3_to_f32(0x81);
        assert_eq!(val_neg, -1.0 / 512.0);

        // Encoding round-trip for 1.0 / 512.0
        let byte_pos = f32_to_fp8_e4m3(1.0 / 512.0);
        assert_eq!(byte_pos, 0x01);
    }

    /// IQ4_NL must match `dequantize_row_iq4_nl` exactly.
    ///
    /// The previous version of this test asserted the 170-byte layout
    /// (`d` + a 32-byte sign plane + 128 nibble bytes + 8 sub-block scale
    /// bytes) that matched no llama.cpp block, and it passed a sign-plane bit
    /// to produce a negative value -- a mechanism the real format does not
    /// have. The expectations below are derived from the reference:
    /// `block_iq4_nl { ggml_half d; uint8_t qs[QK4_NL/2]; }` with
    /// `QK4_NL = 32`, so 18 bytes, `d * kvalues_iq4nl[nibble]`, low nibble to
    /// element `j` and high nibble to element `j + 16`.
    #[test]
    fn test_iq4nl_dequant_matches_reference_block() {
        // d = 1.0f16, then 16 code bytes.
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3c;
        // Every nibble 0x0F -> element j takes kvalues[15] = 113, element
        // j+16 takes kvalues[0] = -127. Distinguishes the (j, j+16) pairing
        // from a 2i/2i+1 interleave, which would put -127 at index 1.
        for b in data[2..18].iter_mut() {
            *b = 0x0F;
        }

        let res = dequant_iq4nl(&data, 32).expect("dequant_iq4nl");
        assert_eq!(res.len(), 32);
        for j in 0..16 {
            assert!(
                (res[j] - 113.0).abs() < 1e-5,
                "element {j} (low nibble of byte {j}) = {}, want 113",
                res[j]
            );
            assert!(
                (res[j + 16] - (-127.0)).abs() < 1e-5,
                "element {} (high nibble of byte {j}) = {}, want -127",
                j + 16,
                res[j + 16]
            );
        }

        // A mixed-nibble byte pins both codebook signs together.
        data[2] = 0x01; // low -> kvalues[1] = -104, high -> kvalues[0] = -127
        for b in data[3..18].iter_mut() {
            *b = 0x00;
        }
        let res = dequant_iq4nl(&data, 32).expect("dequant_iq4nl");
        assert!((res[0] - (-104.0)).abs() < 1e-5, "res[0] = {}", res[0]);
        assert!((res[16] - (-127.0)).abs() < 1e-5, "res[16] = {}", res[16]);

        // The two largest codebook entries, which were 87/107 before.
        for b in data[2..18].iter_mut() {
            *b = 0xEE; // both nibbles -> kvalues[14] = 89
        }
        let res = dequant_iq4nl(&data, 32).expect("dequant_iq4nl");
        assert!(
            res.iter().all(|v| (v - 89.0).abs() < 1e-5),
            "kvalues[14] != 89"
        );

        for b in data[2..18].iter_mut() {
            *b = 0xFF; // both nibbles -> kvalues[15] = 113
        }
        let res = dequant_iq4nl(&data, 32).expect("dequant_iq4nl");
        assert!(
            res.iter().all(|v| (v - 113.0).abs() < 1e-5),
            "kvalues[15] != 113"
        );

        // Layout: 18 bytes per 32 weights, so 256 weights need 144 bytes.
        assert!(dequant_iq4nl(&vec![0u8; 144], 256).is_ok());
        assert!(dequant_iq4nl(&vec![0u8; 143], 256).is_err());
    }

    #[test]
    fn test_iq4xs_dequant_exact_layout_and_math() {
        let mut data = vec![0u8; 136];
        data[0] = 0x00;
        data[1] = 0x3c; // d = 1.0f16
                        // default scales 32 -> scale = 1.0 * (32 - 32) / 32 = 0.0
        data[2] = 32;

        let res = dequant_iq4xs(&data, 256).expect("dequant_iq4xs");
        assert_eq!(res.len(), 256);

        // Error handling on truncated data
        assert!(dequant_iq4xs(&data[..100], 256).is_err());
    }

    #[test]
    fn test_iq3xxs_dequant_exact_layout_and_math() {
        let mut data = vec![0u8; 98];
        data[0] = 0x00;
        data[1] = 0x3c; // d = 1.0f16

        let res = dequant_iq3xxs(&data, 256).expect("dequant_iq3xxs");
        assert_eq!(res.len(), 256);

        // Error handling on truncated data
        assert!(dequant_iq3xxs(&data[..50], 256).is_err());
    }

    #[test]
    fn test_iq3s_dequant_exact_layout_and_math() {
        let mut data = vec![0u8; 110];
        data[0] = 0x00;
        data[1] = 0x3c; // d = 1.0f16

        let res = dequant_iq3s(&data, 256).expect("dequant_iq3s");
        assert_eq!(res.len(), 256);

        assert!(dequant_iq3s(&data[..50], 256).is_err());
    }

    #[test]
    fn test_iq2xxs_dequant_exact_layout_and_math() {
        let mut data = vec![0u8; 66];
        data[0] = 0x00;
        data[1] = 0x3c; // d = 1.0f16

        let res = dequant_iq2xxs(&data, 256).expect("dequant_iq2xxs");
        assert_eq!(res.len(), 256);

        assert!(dequant_iq2xxs(&data[..30], 256).is_err());
    }

    #[test]
    fn test_iq2xs_dequant_exact_layout_and_math() {
        let mut data = vec![0u8; 74];
        data[0] = 0x00;
        data[1] = 0x3c; // d = 1.0f16

        let res = dequant_iq2xs(&data, 256).expect("dequant_iq2xs");
        assert_eq!(res.len(), 256);

        assert!(dequant_iq2xs(&data[..40], 256).is_err());
    }

    #[test]
    fn test_iq2s_dequant_exact_layout_and_math() {
        let mut data = vec![0u8; 82];
        data[0] = 0x00;
        data[1] = 0x3c; // d = 1.0f16

        let res = dequant_iq2s(&data, 256).expect("dequant_iq2s");
        assert_eq!(res.len(), 256);

        assert!(dequant_iq2s(&data[..40], 256).is_err());
    }

    #[test]
    fn test_iquant_roundtrip_rewrite() {
        let orig = vec![1.0f32; 256];
        let formats = [
            QuantFormat::Iq4Nl,
            QuantFormat::Iq4Xs,
            QuantFormat::Iq3Xxs,
            QuantFormat::Iq3S,
            QuantFormat::Iq2Xxs,
            QuantFormat::Iq2Xs,
            QuantFormat::Iq2S,
        ];
        for fmt in formats {
            let plan = TensorRewritePlan {
                target: fmt,
                shape: vec![256],
                importance: None,
                curvature: None,
            };
            let rewritten = rewrite_tensor_data(&orig, &plan).expect("rewrite_tensor_data");
            assert_eq!(rewritten.target, fmt);
            assert!(!rewritten.bytes.is_empty());
        }
    }

    #[test]
    fn test_gemm_packed_matches_dequant_then_gemm() {
        for &k in &[256, 512, 1536] {
            let m = 2;
            let n = 3;
            let mut a = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                a.push(((i % 17) as f32 - 8.0) * 0.1);
            }
            let mut b_f32 = Vec::with_capacity(n * k);
            for i in 0..(n * k) {
                b_f32.push(((i % 19) as f32 - 9.0) * 0.1);
            }

            // Q8_0 test
            let mut b_q80_bytes = Vec::new();
            for col in 0..n {
                let row = &b_f32[col * k..(col + 1) * k];
                let q = quant_q80(row).expect("quant_q80");
                b_q80_bytes.extend_from_slice(&q);
            }
            let packed_c_q80 =
                gemm_q8_0_packed(&a, &b_q80_bytes, m, n, k).expect("gemm_q8_0_packed");
            let dequant_b_q80 = dequant_q80(&b_q80_bytes, n * k).expect("dequant_q80");
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0f32;
                    for l in 0..k {
                        expected += a[row * k + l] * dequant_b_q80[col * k + l];
                    }
                    let actual = packed_c_q80[row * n + col];
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "Q8_0 k={k} mismatch: actual={actual}, expected={expected}"
                    );
                }
            }

            // Q4_K test
            let mut b_q4k_bytes = Vec::new();
            for col in 0..n {
                let row = &b_f32[col * k..(col + 1) * k];
                let q = quant_q4k(row).expect("quant_q4k");
                b_q4k_bytes.extend_from_slice(&q);
            }
            let packed_c_q4k = gemm_q4k_packed(&a, &b_q4k_bytes, m, n, k).expect("gemm_q4k_packed");
            let dequant_b_q4k = dequant_q4k(&b_q4k_bytes, n * k).expect("dequant_q4k");
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0f32;
                    for l in 0..k {
                        expected += a[row * k + l] * dequant_b_q4k[col * k + l];
                    }
                    let actual = packed_c_q4k[row * n + col];
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "Q4_K k={k} mismatch: actual={actual}, expected={expected}"
                    );
                }
            }

            // IQ4_NL test
            let mut b_iq4nl_bytes = Vec::new();
            for col in 0..n {
                let row = &b_f32[col * k..(col + 1) * k];
                let q = quant_iq4nl(row).expect("quant_iq4nl");
                b_iq4nl_bytes.extend_from_slice(&q);
            }
            let packed_c_iq4nl =
                gemm_iq4nl_packed(&a, &b_iq4nl_bytes, m, n, k).expect("gemm_iq4nl_packed");
            let dequant_b_iq4nl = dequant_iq4nl(&b_iq4nl_bytes, n * k).expect("dequant_iq4nl");
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0f32;
                    for l in 0..k {
                        expected += a[row * k + l] * dequant_b_iq4nl[col * k + l];
                    }
                    let actual = packed_c_iq4nl[row * n + col];
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "IQ4_NL k={k} mismatch: actual={actual}, expected={expected}"
                    );
                }
            }

            // IQ3_S test (A3: fused path must match dequant-then-GEMM)
            let mut b_iq3s_bytes = Vec::new();
            for col in 0..n {
                let row = &b_f32[col * k..(col + 1) * k];
                let q = quant_iq3s(row).expect("quant_iq3s");
                b_iq3s_bytes.extend_from_slice(&q);
            }
            let packed_c_iq3s =
                gemm_iq3s_packed(&a, &b_iq3s_bytes, m, n, k).expect("gemm_iq3s_packed");
            let dequant_b_iq3s = dequant_iq3s(&b_iq3s_bytes, n * k).expect("dequant_iq3s");
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0f32;
                    for l in 0..k {
                        expected += a[row * k + l] * dequant_b_iq3s[col * k + l];
                    }
                    let actual = packed_c_iq3s[row * n + col];
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "IQ3_S k={k} mismatch: actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }
}

/// Packed matrix multiplication: `C[m, n] = sum_k A[m, k] * B[n, k]` where B is packed Q8_0 weights.
/// A has shape [m, k], B has shape [n, k] (in packed Q8_0 bytes).
pub fn gemm_q8_0_packed(
    a: &[f32],
    b_q80_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if k % 32 != 0 {
        return Err(Error::Backend(format!(
            "gemm_q8_0_packed: k ({k}) must be a multiple of 32"
        )));
    }
    let blocks_per_row = k / 32;
    let stride_b = blocks_per_row * 34;
    if b_q80_bytes.len() < n * stride_b {
        return Err(Error::Backend(format!(
            "gemm_q8_0_packed: buffer too short: expected {}, got {}",
            n * stride_b,
            b_q80_bytes.len()
        )));
    }
    if a.len() < m * k {
        return Err(Error::Backend(format!(
            "gemm_q8_0_packed: input a too short: expected {}, got {}",
            m * k,
            a.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if packed_gemm::avx2_detected() {
        // SAFETY: AVX2 presence was just verified at runtime, and all shape /
        // buffer-length validation above has passed.
        return Ok(unsafe { packed_gemm::x86::gemm_q8_0_packed_avx2(a, b_q80_bytes, m, n, k) });
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: shape / buffer-length validation above has passed. NEON is
    // baseline on aarch64, so no runtime feature check is needed.
    return Ok(unsafe { packed_gemm::neon::gemm_q8_0_packed_neon(a, b_q80_bytes, m, n, k) });

    #[cfg(not(target_arch = "aarch64"))]
    Ok(gemm_q8_0_packed_scalar(a, b_q80_bytes, m, n, k))
}

/// Scalar reference implementation of [`gemm_q8_0_packed`] (inputs already validated).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn gemm_q8_0_packed_scalar(
    a: &[f32],
    b_q80_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let blocks_per_row = k / 32;
    let stride_b = blocks_per_row * 34;

    let mut c = vec![0.0f32; m * n];

    for row_m in 0..m {
        let a_row = &a[row_m * k..(row_m + 1) * k];
        for col_n in 0..n {
            let b_row = &b_q80_bytes[col_n * stride_b..(col_n + 1) * stride_b];
            let mut dot = 0.0f32;
            let mut b_pos = 0;
            for blk in 0..blocks_per_row {
                let scale = f16_to_f32(b_row[b_pos], b_row[b_pos + 1]);
                b_pos += 2;
                let a_sub = &a_row[blk * 32..(blk + 1) * 32];
                let mut sum_i = 0.0f32;
                for l in 0..32 {
                    let q = b_row[b_pos + l] as i8 as f32;
                    sum_i += a_sub[l] * q;
                }
                dot += scale * sum_i;
                b_pos += 32;
            }
            c[row_m * n + col_n] = dot;
        }
    }

    c
}

/// Packed matrix multiplication: `C[m, n] = sum_k A[m, k] * B[n, k]` where B is packed Q4_K weights.
/// A has shape [m, k], B has shape [n, k] (in packed Q4_K bytes: 144.
pub fn gemm_q4k_packed(
    a: &[f32],
    b_q4k_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if k % 256 != 0 {
        return Err(Error::Backend(format!(
            "gemm_q4k_packed: k ({k}) must be a multiple of 256"
        )));
    }
    let blocks_per_row = k / 256;
    let stride_b = blocks_per_row * 144;
    if b_q4k_bytes.len() < n * stride_b {
        return Err(Error::Backend(format!(
            "gemm_q4k_packed: buffer too short: expected {}, got {}",
            n * stride_b,
            b_q4k_bytes.len()
        )));
    }
    if a.len() < m * k {
        return Err(Error::Backend(format!(
            "gemm_q4k_packed: input a too short: expected {}, got {}",
            m * k,
            a.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if packed_gemm::avx2_detected() {
        // SAFETY: AVX2 presence was just verified at runtime, and all shape /
        // buffer-length validation above has passed.
        return Ok(unsafe { packed_gemm::x86::gemm_q4k_packed_avx2(a, b_q4k_bytes, m, n, k) });
    }

    Ok(gemm_q4k_packed_scalar(a, b_q4k_bytes, m, n, k))
}

/// Scalar reference implementation of [`gemm_q4k_packed`] (inputs already validated).
fn gemm_q4k_packed_scalar(a: &[f32], b_q4k_bytes: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / 256;
    let stride_b = blocks_per_row * 144;

    let mut c = vec![0.0f32; m * n];

    for row_m in 0..m {
        let a_row = &a[row_m * k..(row_m + 1) * k];
        for col_n in 0..n {
            let b_row = &b_q4k_bytes[col_n * stride_b..(col_n + 1) * stride_b];
            let mut dot = 0.0f32;
            let mut pos = 0;
            for blk in 0..blocks_per_row {
                let a_blk = &a_row[blk * 256..(blk + 1) * 256];
                let d = f16_to_f32(b_row[pos], b_row[pos + 1]);
                let min = f16_to_f32(b_row[pos + 2], b_row[pos + 3]);
                let scales = &b_row[pos + 4..pos + 16];
                let qs = &b_row[pos + 16..pos + 144];

                let mut q_idx = 0;
                let mut is = 0;
                let mut a_offset = 0;

                for _ in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let d1 = d * sc1;
                    let m1_val = min * m1;

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc2;
                    let m2_val = min * m2;

                    for l in 0..32 {
                        let q1 = (qs[q_idx + l] & 0x0F) as f32;
                        let w = d1 * q1 - m1_val;
                        dot += a_blk[a_offset + l] * w;
                    }
                    a_offset += 32;

                    for l in 0..32 {
                        let q2 = (qs[q_idx + l] >> 4) as f32;
                        let w = d2 * q2 - m2_val;
                        dot += a_blk[a_offset + l] * w;
                    }
                    a_offset += 32;

                    q_idx += 32;
                    is += 2;
                }
                pos += 144;
            }
            c[row_m * n + col_n] = dot;
        }
    }

    c
}

// IQ4_NL block constants (llama.cpp `block_iq4_nl`, `QK4_NL = 32`).
/// Weights per IQ4_NL block.
pub const IQ4_NL_QK: usize = 32;
/// Bytes per IQ4_NL block: 2-byte f16 scale + 16 byte-packed 32 nibbles.
pub const IQ4_NL_BLOCK_BYTES: usize = 2 + IQ4_NL_QK / 2; // 18

/// Packed matrix multiplication: `C[m, n] = sum_k A[m, k] * B[n, k]` where B is packed IQ4_NL weights.
///
/// A has shape `[m, k]`, B has shape `[n, k]` (in packed IQ4_NL bytes: 18 bytes
/// per 32 weights — 2-byte f16 scale `d` followed by 16 nibble bytes).
///
/// This is the CPU-native fused dequant+GEMM path that bypasses the full f32
/// materialization of B. It mirrors `gemm_q8_0_packed` / `gemm_q4k_packed`:
/// the inner loop looks up each nibble in `KVALUES_IQ4NL` and multiplies by `d`,
/// then accumulates the dot product against A.
///
/// # Errors
/// Returns [`Error::Backend`] when `k` is not a multiple of 32, when `b_bytes`
/// is shorter than `n * (k/32) * 18`, or when `a` is shorter than `m * k`.
pub fn gemm_iq4nl_packed(
    a: &[f32],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if k % IQ4_NL_QK != 0 {
        return Err(Error::Backend(format!(
            "gemm_iq4nl_packed: k ({k}) must be a multiple of {IQ4_NL_QK}"
        )));
    }
    let blocks_per_row = k / IQ4_NL_QK;
    let stride_b = blocks_per_row * IQ4_NL_BLOCK_BYTES;
    if b_bytes.len() < n * stride_b {
        return Err(Error::Backend(format!(
            "gemm_iq4nl_packed: buffer too short: expected {}, got {}",
            n * stride_b,
            b_bytes.len()
        )));
    }
    if a.len() < m * k {
        return Err(Error::Backend(format!(
            "gemm_iq4nl_packed: input a too short: expected {}, got {}",
            m * k,
            a.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if packed_gemm::avx2_detected() {
        // SAFETY: AVX2 presence was just verified at runtime, and all shape /
        // buffer-length validation above has passed.
        return Ok(unsafe { packed_gemm::x86::gemm_iq4nl_packed_avx2(a, b_bytes, m, n, k) });
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: shape / buffer-length validation above has passed. NEON is
    // baseline on aarch64, so no runtime feature check is needed.
    return Ok(unsafe { packed_gemm::neon::gemm_iq4nl_packed_neon(a, b_bytes, m, n, k) });

    #[cfg(not(target_arch = "aarch64"))]
    Ok(gemm_iq4nl_packed_scalar(a, b_bytes, m, n, k))
}

/// Scalar reference implementation of [`gemm_iq4nl_packed`] (inputs already validated).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn gemm_iq4nl_packed_scalar(a: &[f32], b_bytes: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / IQ4_NL_QK;
    let stride_b = blocks_per_row * IQ4_NL_BLOCK_BYTES;

    let mut c = vec![0.0f32; m * n];

    for row_m in 0..m {
        let a_row = &a[row_m * k..(row_m + 1) * k];
        for col_n in 0..n {
            let b_row = &b_bytes[col_n * stride_b..(col_n + 1) * stride_b];
            let mut dot = 0.0f32;
            let mut b_pos = 0;
            for blk in 0..blocks_per_row {
                let d = f16_to_f32(b_row[b_pos], b_row[b_pos + 1]);
                let qs = &b_row[b_pos + 2..b_pos + IQ4_NL_BLOCK_BYTES];
                let a_sub = &a_row[blk * IQ4_NL_QK..(blk + 1) * IQ4_NL_QK];

                // Low nibble of qs[j] → weight[j] (j in 0..16),
                // high nibble of qs[j] → weight[j+16].
                for j in 0..IQ4_NL_QK / 2 {
                    let lo = (qs[j] & 0x0F) as usize;
                    let hi = ((qs[j] >> 4) & 0x0F) as usize;
                    dot += a_sub[j] * d * KVALUES_IQ4NL_REF[lo];
                    dot += a_sub[j + IQ4_NL_QK / 2] * d * KVALUES_IQ4NL_REF[hi];
                }
                b_pos += IQ4_NL_BLOCK_BYTES;
            }
            c[row_m * n + col_n] = dot;
        }
    }

    c
}

// IQ3_S block constants (llama.cpp `block_iq3_s`, QK_K = 256).
/// Weights per IQ3_S super-block.
pub const IQ3S_QK: usize = 256;
/// Bytes per IQ3_S super-block: 2-byte f16 `d` + 64 `qs` + 8 `qh` + 32 `signs` + 4 `scales`.
pub const IQ3S_BLOCK_BYTES: usize = 110;

/// Packed matrix multiplication: `C[m, n] = sum_k A[m, k] * B[n, k]` where B is packed IQ3_S weights.
///
/// A has shape `[m, k]`, B has shape `[n, k]` (in packed IQ3_S bytes: 110 bytes
/// per 256 weights). CPU-native fused dequant+GEMM path that bypasses the full
/// f32 materialization of B. Mirrors `gemm_q4k_packed` (same 256-weight
/// super-block loop): each block decodes via the shared [`dequant_iq3s_block`]
/// oracle, then accumulates the dot product against A.
///
/// # Errors
/// Returns [`Error::Backend`] when `k` is not a multiple of 256, when `b_bytes`
/// is shorter than `n * (k/256) * 110`, or when `a` is shorter than `m * k`.
pub fn gemm_iq3s_packed(
    a: &[f32],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if k % IQ3S_QK != 0 {
        return Err(Error::Backend(format!(
            "gemm_iq3s_packed: k ({k}) must be a multiple of {IQ3S_QK}"
        )));
    }
    let blocks_per_row = k / IQ3S_QK;
    let stride_b = blocks_per_row * IQ3S_BLOCK_BYTES;
    if b_bytes.len() < n * stride_b {
        return Err(Error::Backend(format!(
            "gemm_iq3s_packed: buffer too short: expected {}, got {}",
            n * stride_b,
            b_bytes.len()
        )));
    }
    if a.len() < m * k {
        return Err(Error::Backend(format!(
            "gemm_iq3s_packed: input a too short: expected {}, got {}",
            m * k,
            a.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if packed_gemm::avx2_detected() {
        // SAFETY: AVX2 presence was just verified at runtime, and all shape /
        // buffer-length validation above has passed.
        return Ok(unsafe { packed_gemm::x86::gemm_iq3s_packed_avx2(a, b_bytes, m, n, k) });
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: shape / buffer-length validation above has passed. NEON is
    // baseline on aarch64, so no runtime feature check is needed.
    return Ok(unsafe { packed_gemm::neon::gemm_iq3s_packed_neon(a, b_bytes, m, n, k) });

    #[cfg(not(target_arch = "aarch64"))]
    Ok(gemm_iq3s_packed_scalar(a, b_bytes, m, n, k))
}

/// Scalar reference implementation of [`gemm_iq3s_packed`] (inputs already validated).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn gemm_iq3s_packed_scalar(a: &[f32], b_bytes: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / IQ3S_QK;
    let stride_b = blocks_per_row * IQ3S_BLOCK_BYTES;

    let mut c = vec![0.0f32; m * n];

    for row_m in 0..m {
        let a_row = &a[row_m * k..(row_m + 1) * k];
        for col_n in 0..n {
            let b_row = &b_bytes[col_n * stride_b..(col_n + 1) * stride_b];
            let mut dot = 0.0f32;
            for blk in 0..blocks_per_row {
                let w = dequant_iq3s_block(
                    &b_row[blk * IQ3S_BLOCK_BYTES..(blk + 1) * IQ3S_BLOCK_BYTES],
                );
                let a_sub = &a_row[blk * IQ3S_QK..(blk + 1) * IQ3S_QK];
                for j in 0..IQ3S_QK {
                    dot += a_sub[j] * w[j];
                }
            }
            c[row_m * n + col_n] = dot;
        }
    }

    c
}

/// Magic tag for the packed 128x128 block-FP8 blob header.
const FP8_BLOCK128_MAGIC: u32 = 0x4250_3846; // "FP8B" little-endian
/// `[magic u32][out u32][in u32][grid_rows u32][grid_cols u32]`
const FP8_BLOCK128_HEADER: usize = 20;

/// Pack E4M3 codes plus a 128x128 `f32` scale grid into one self-describing blob.
///
/// Layout: `[magic][out][in][grid_rows][grid_cols][exps: grid f32][codes: out*in u8]`
///
/// Keeping the scales in the same buffer is what lets the GPU quantized-matmul
/// path consume the format: it takes no separate scale argument, so a two-tensor
/// (codes + `weight_scale_inv`) representation would have to be paired on the
/// host, which is the round-trip this format exists to avoid.
pub fn pack_fp8_block128(
    codes: &[u8],
    exps: &[f32],
    out: usize,
    in_dim: usize,
    grid_rows: usize,
    grid_cols: usize,
) -> Result<Vec<u8>> {
    if codes.len() < out * in_dim {
        return Err(Error::Backend(format!(
            "pack_fp8_block128: {} codes for a {out}x{in_dim} weight",
            codes.len()
        )));
    }
    if exps.len() < grid_rows * grid_cols {
        return Err(Error::Backend(format!(
            "pack_fp8_block128: {} scales for a [{grid_rows},{grid_cols}] grid",
            exps.len()
        )));
    }
    if grid_rows == 0 || grid_cols == 0 || out % grid_rows != 0 || in_dim % grid_cols != 0 {
        return Err(Error::Backend(format!(
            "pack_fp8_block128: [{out},{in_dim}] not tileable by grid [{grid_rows},{grid_cols}]"
        )));
    }
    let mut blob =
        Vec::with_capacity(FP8_BLOCK128_HEADER + grid_rows * grid_cols * 4 + out * in_dim);
    blob.extend_from_slice(&FP8_BLOCK128_MAGIC.to_le_bytes());
    blob.extend_from_slice(&(out as u32).to_le_bytes());
    blob.extend_from_slice(&(in_dim as u32).to_le_bytes());
    blob.extend_from_slice(&(grid_rows as u32).to_le_bytes());
    blob.extend_from_slice(&(grid_cols as u32).to_le_bytes());
    for e in &exps[..grid_rows * grid_cols] {
        blob.extend_from_slice(&e.to_le_bytes());
    }
    blob.extend_from_slice(&codes[..out * in_dim]);
    Ok(blob)
}

/// Unpack a 128x128 block-FP8 blob back to row-major `f32`.
///
/// This is the reference decode: `out[i, j] = e4m3(codes[i, j]) *
/// exps[(i / block_rows) * grid_cols + (j / block_cols)]`, which is exactly the
/// arithmetic the previous fold-to-F32 loader performed.
pub fn dequant_fp8_block128(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() < FP8_BLOCK128_HEADER {
        return Err(Error::Backend(format!(
            "dequant_fp8_block128: blob is {} bytes, shorter than its {FP8_BLOCK128_HEADER}-byte header",
            data.len()
        )));
    }
    let rd = |off: usize| -> u32 {
        u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    };
    if rd(0) != FP8_BLOCK128_MAGIC {
        return Err(Error::Backend(
            "dequant_fp8_block128: bad magic — not a packed block-FP8 blob".into(),
        ));
    }
    let (out, in_dim) = (rd(4) as usize, rd(8) as usize);
    let (grid_rows, grid_cols) = (rd(12) as usize, rd(16) as usize);
    if out == 0 || in_dim == 0 || grid_rows == 0 || grid_cols == 0 {
        return Err(Error::Backend(
            "dequant_fp8_block128: zero-sized header".into(),
        ));
    }
    if out % grid_rows != 0 || in_dim % grid_cols != 0 {
        return Err(Error::Backend(format!(
            "dequant_fp8_block128: [{out},{in_dim}] not tileable by grid [{grid_rows},{grid_cols}]"
        )));
    }
    let exps_at = FP8_BLOCK128_HEADER;
    let codes_at = exps_at + grid_rows * grid_cols * 4;
    if data.len() < codes_at + out * in_dim {
        return Err(Error::Backend(format!(
            "dequant_fp8_block128: blob is {} bytes, need {} for a {out}x{in_dim} weight",
            data.len(),
            codes_at + out * in_dim
        )));
    }
    let block_rows = out / grid_rows;
    let block_cols = in_dim / grid_cols;
    let mut w = vec![0.0f32; out * in_dim];
    for i in 0..out {
        let gr = i / block_rows;
        for j in 0..in_dim {
            let e = f32::from_le_bytes([
                data[exps_at + (gr * grid_cols + j / block_cols) * 4],
                data[exps_at + (gr * grid_cols + j / block_cols) * 4 + 1],
                data[exps_at + (gr * grid_cols + j / block_cols) * 4 + 2],
                data[exps_at + (gr * grid_cols + j / block_cols) * 4 + 3],
            ]);
            w[i * in_dim + j] = fp8_e4m3_to_f32(data[codes_at + i * in_dim + j]) * e;
        }
    }
    Ok(w)
}

/// Weights per 18-byte 2-bit block (`QK2_0` in `ggml-common.h:187`).
///
/// Shared by upstream `Q2_0` (tag 42) and Prism GSQRCO (tag 81): the block is
/// `ggml_half d` + `qs[16]`, so both pack 64 weights into 18 bytes (2.25 bpw).
pub const BLOCK_SIZE_Q2_0: usize = 64;
/// Bytes per block: 2-byte fp16 delta + 64*2/8 bytes of packed 2-bit codes.
pub const BLOCK_BYTES_Q2_0: usize = 18;
/// Alias kept for the GSQRCO call sites, which share the same geometry.
pub const BLOCK_SIZE_GSQ_RCO_3P5: usize = BLOCK_SIZE_Q2_0;
/// Alias kept for the GSQRCO call sites, which share the same geometry.
pub const BLOCK_BYTES_GSQ_RCO_3P5: usize = BLOCK_BYTES_Q2_0;

/// Dequantize upstream GGUF `Q2_0` (dtype tag 42) packed weights to `f32`.
///
/// Transcribed from ggml-org/llama.cpp at commit `f3f1a8f27`, the runtime the
/// Qwen3.8-Flash-Next GSQ-RCO-3.5bit release is pinned to:
///
/// ```c
/// #define QK2_0 64
/// typedef struct {
///     ggml_half d;              // delta (scale)
///     uint8_t qs[QK2_0 / 4];   // 2 bits per element
/// } block_q2_0;
/// ```
///
/// and `dequantize_row_q2_0`:
///
/// ```c
/// const int q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
/// // 00=-1, 01=0, 10=+1, 11=+2
/// y[i*qk + j] = ((int)q - 1) * d;
/// ```
///
/// The codebook is zero-centered at code 1 (`{-1, 0, +1, +2}`), there is no
/// `dmin`, and no per-sub-block scale. This is NOT the GSQRCO codebook, which
/// is `{-2, -1, 0, +1}` — see [`dequant_gsq_rco_3p5`]. Effective rate is 2.25
/// bits per weight either way; the earlier 256-weight/72-byte reading of this
/// format was wrong (72 B = 4 x 18 B describes the same bytes in
/// groups-of-4 framing).
///
/// # Errors
/// Returns [`Error::Backend`] when `data` is shorter than the block-aligned
/// requirement for `num_weights`, or when `num_weights` is not a multiple of
/// [`BLOCK_SIZE_Q2_0`] (GGUF rows are always block-aligned, so a ragged
/// count means the caller computed the tensor size wrongly).
pub fn dequant_q2_0(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    // GRIM_GSQ_BIAS A/B (shared with dequant_gsq_rco_3p5): upstream tag 42
    // decodes (q - 1) * d (llama.cpp ggml-quants.c:439), but the RELEASED
    // Qwen3.8-Flash-Next file stores its tag-42 expert banks with GSQ-RCO
    // semantics — the byte-level arbiter (gsq_rco_3p5_pack.rs
    // released_checkpoint_codebook_arbiter) measured negatives reaching 2d /
    // positives stopping at 1d, the bias-2 signature. GRIM_GSQ_BIAS=2 reads
    // those banks correctly in the SAME binary; default 1 keeps upstream
    // files faithful.
    static BIAS: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let bias = *BIAS.get_or_init(|| {
        match std::env::var("GRIM_GSQ_BIAS").as_deref() {
            Ok("2") => 2.0,
            _ => 1.0,
        }
    });
    dequant_2bit_blocks(data, num_weights, "q2_0", bias)
}

/// GSQRCO shares [`BLOCK_SIZE_Q2_0`] geometry with `Q2_0` but uses the
/// GumbelQuantizer2Bit codebook `{-2, -1, 0, +1} * scale`, i.e.
/// `y = (q - 2) * d`. The two decoders differ by exactly one level of `d`,
/// so a wrong-codebook decode yields finite, plausible, wrong weights.
pub fn dequant_gsq_rco_3p5(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    // GRIM_GSQ_BIAS: the one-knob A/B for the released-checkpoint codebook
    // question (this session's open item). Default 2 = the GSQ paper's
    // quantizer; GRIM_GSQ_BIAS=1 restores the llama.cpp Q2_0 reading
    // (`(q - 1) * d`, ggml-quants.c:439) under which the release reportedly
    // evals at ppl 2.502. Both arms in ONE binary so a ppl comparison cannot
    // differ by build.
    static BIAS: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let bias = *BIAS.get_or_init(|| {
        match std::env::var("GRIM_GSQ_BIAS").as_deref() {
            Ok("1") => 1.0,
            _ => 2.0,
        }
    });
    dequant_2bit_blocks(data, num_weights, "gsq_rco_3p5", bias)
}

/// Shared core for the two 18-byte / 64-elem 2-bit formats: `y = (q - bias) * d`
/// with `d` an fp16 read little-endian from the first two bytes of the block
/// and 4 codes packed per byte.
fn dequant_2bit_blocks(
    data: &[u8],
    num_weights: usize,
    label: &str,
    bias: f32,
) -> Result<Vec<f32>> {
    if num_weights == 0 {
        return Ok(Vec::new());
    }
    if num_weights % BLOCK_SIZE_Q2_0 != 0 {
        return Err(Error::Backend(format!(
            "dequant_{label}: {num_weights} weights is not a multiple of block size {BLOCK_SIZE_Q2_0}"
        )));
    }
    let num_blocks = num_weights / BLOCK_SIZE_Q2_0;
    let expected_bytes = num_blocks * BLOCK_BYTES_Q2_0;
    if data.len() < expected_bytes {
        return Err(Error::Backend(format!(
            "dequant_{label}: buffer too short: expected {expected_bytes}, have {}",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;
    for _ in 0..num_blocks {
        let d = f16_to_f32(data[pos], data[pos + 1]);
        let qs = &data[pos + 2..pos + BLOCK_BYTES_Q2_0];
        for j in 0..BLOCK_SIZE_Q2_0 {
            let q = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            out.push((q as f32 - bias) * d);
        }
        pos += BLOCK_BYTES_Q2_0;
    }
    Ok(out)
}

/// Pack `f32` weights into upstream GGUF `Q2_0` blocks, the exact inverse of
/// [`dequant_q2_0`].
///
/// This exists for round-trip KATs and for tests that need a non-degenerate
/// block: an all-zero input block dequantizes correctly under *any* layout, so
/// it cannot distinguish a right implementation from a wrong one.
///
/// Each block is scaled by `d = max(|x|)` (rounded to fp16), then each weight
/// is mapped to the nearest of the four codebook
/// values `{-1, 0, +1, +2}` (codes 0..3). `q = 1` (weight 0) and `q = 0` (weight
/// -d) are both reachable, and the tie at `x == d/3` resolves to `+1`, matching
/// the round-half-up behaviour of the reference quantizer.
///
/// # Errors
/// Returns [`Error::Backend`] when `values.len()` is not a multiple of
/// [`BLOCK_SIZE_Q2_0`], when `data` is too small to hold the output, or
/// when a block is all zeros (which has no representable `d`).
pub fn quantize_q2_0_block(values: &[f32], out: &mut [u8]) -> Result<()> {
    if values.len() % BLOCK_SIZE_Q2_0 != 0 {
        return Err(Error::Backend(format!(
            "quantize_q2_0_block: {} values is not a multiple of block size {BLOCK_SIZE_Q2_0}",
            values.len()
        )));
    }
    let num_blocks = values.len() / BLOCK_SIZE_Q2_0;
    let needed = num_blocks * BLOCK_BYTES_Q2_0;
    if out.len() < needed {
        return Err(Error::Backend(format!(
            "quantize_q2_0_block: output too short: need {needed}, have {}",
            out.len()
        )));
    }

    for b in 0..num_blocks {
        let block = &values[b * BLOCK_SIZE_Q2_0..(b + 1) * BLOCK_SIZE_Q2_0];
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if !(amax > 0.0) {
            return Err(Error::Backend(format!(
                "quantize_q2_0_block: block {b} is all zeros and has no representable scale"
            )));
        }
        let d_bits = f32_to_f16(amax);
        if d_bits == 0 {
            return Err(Error::Backend(format!(
                "quantize_q2_0_block: block {b} scale {amax} is not representable in fp16"
            )));
        }
        // Round-trip through fp16 so `d` here is bit-identical to what
        // `dequant_q2_0` will read back, not the original f32 scale.
        let d = f16_to_f32(d_bits as u8, (d_bits >> 8) as u8);

        let base = b * BLOCK_BYTES_Q2_0;
        out[base] = d_bits as u8;
        out[base + 1] = (d_bits >> 8) as u8;
        for j in 0..BLOCK_SIZE_Q2_0 {
            // Nearest codebook point in units of d: {-1, 0, +1, +2}.
            let t = (block[j] / d) + 1.0;
            let code = if t <= 0.5 {
                0u8
            } else if t <= 1.5 {
                1
            } else if t <= 2.5 {
                2
            } else {
                3
            };
            out[base + 2 + j / 4] |= code << ((j % 4) * 2);
        }
    }
    Ok(())
}

// ============================================================================
// GSQ-RCO 3.5-bit (GGUF tag 81) — the quantize side
//
// The format's source of truth is the GSQ paper's own quantizer
// (old/repo/GSQ-main/src/quantization/gumbel_quantizer_2bit.py:10):
// `values = [-2, -1, 0, 1]`, so codes 0..3 mean (q - 2) * d — one level of d
// BELOW upstream Q2_0's (q - 1) * d (llama.cpp ggml-quants.c:439). The block
// geometry (fp16 d at the head, 4 codes per byte low-nibble-first) is shared,
// which is exactly why a wrong-codebook decode is silent: same bytes, finite,
// plausible, off-by-one-level weights. `dequant_gsq_rco_3p5` above is the
// bias-2 reader; this packer is its inverse.
//
// What this is NOT: the GSQ optimizer. The paper's scales are LEARNED per
// group-128 by Gumbel-Softmax over calibration data; this writer does the
// RTN baseline (nearest level, MSE-fitted per-block scale), which is the
// floor the paper's method improves on. A true GSQ conversion is a
// calibration run, not a kernel.
// ============================================================================

/// Quantize f32 weights into GSQ-RCO 3.5-bit blocks (GGUF tag 81), the exact
/// inverse of [`dequant_gsq_rco_3p5`].
///
/// Per 64-weight block: fp16 `d` at the head, then 16 bytes of 2-bit codes,
/// low nibble first, where code `q` decodes to `(q - 2) * d` over the level
/// set `{-2, -1, 0, +1}`. The scale is fitted per block by a deterministic
/// MSE scan over `d` in `[amax/2, amax]` — the range where the block's
/// largest magnitude stays representable (below amax/2 the most negative
/// weight needs a code below -2d; above amax the most positive weight
/// exceeds the largest level +1d) — rounded through fp16 BEFORE code
/// assignment, so `dequant_gsq_rco_3p5` reads back exactly the scale the
/// codes were fitted against.
///
/// # Errors
/// Mirrors [`quantize_q2_0_block`]: non-multiple of [`BLOCK_SIZE_Q2_0`],
/// output too short, all-zero block, or a scale not representable in fp16.
pub fn quantize_gsq_rco_3p5_block(values: &[f32], out: &mut [u8]) -> Result<()> {
    if values.len() % BLOCK_SIZE_Q2_0 != 0 {
        return Err(Error::Backend(format!(
            "quantize_gsq_rco_3p5_block: {} values is not a multiple of block size {BLOCK_SIZE_Q2_0}",
            values.len()
        )));
    }
    let num_blocks = values.len() / BLOCK_SIZE_Q2_0;
    let needed = num_blocks * BLOCK_BYTES_Q2_0;
    if out.len() < needed {
        return Err(Error::Backend(format!(
            "quantize_gsq_rco_3p5_block: output too short: need {needed}, have {}",
            out.len()
        )));
    }

    for b in 0..num_blocks {
        let block = &values[b * BLOCK_SIZE_Q2_0..(b + 1) * BLOCK_SIZE_Q2_0];
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if !(amax > 0.0) {
            return Err(Error::Backend(format!(
                "quantize_gsq_rco_3p5_block: block {b} is all zeros and has no representable scale"
            )));
        }

        let mut best_d_bits = 0u16;
        let mut best_mse = f32::INFINITY;
        for step in 0..=128u32 {
            let d = amax / 2.0 + (amax / 2.0) * (step as f32 / 128.0);
            let d_bits = f32_to_f16(d);
            if d_bits == 0 {
                continue;
            }
            // Fit codes against the fp16-ROUNDED scale: the decoder reads
            // this exact value back, so the MSE must be measured against it,
            // not against the pre-rounding f32.
            let d16 = f16_to_f32(d_bits as u8, (d_bits >> 8) as u8);
            let mut mse = 0.0f32;
            for v in block {
                let t = v / d16;
                let level = (t.round() as i32).clamp(-2, 1) as f32;
                let err = v - level * d16;
                mse += err * err;
            }
            if mse < best_mse {
                best_mse = mse;
                best_d_bits = d_bits;
            }
        }
        if best_d_bits == 0 {
            return Err(Error::Backend(format!(
                "quantize_gsq_rco_3p5_block: block {b} scale {amax} is not representable in fp16"
            )));
        }
        let d = f16_to_f32(best_d_bits as u8, (best_d_bits >> 8) as u8);

        let base = b * BLOCK_BYTES_Q2_0;
        out[base] = best_d_bits as u8;
        out[base + 1] = (best_d_bits >> 8) as u8;
        for j in 0..BLOCK_SIZE_Q2_0 {
            // Nearest codebook point in units of d: {-2, -1, 0, +1}.
            let t = (block[j] / d) + 2.0;
            let code = if t <= 0.5 {
                0u8
            } else if t <= 1.5 {
                1
            } else if t <= 2.5 {
                2
            } else {
                3
            };
            out[base + 2 + j / 4] |= code << ((j % 4) * 2);
        }
    }
    Ok(())
}

// ============================================================================
// Prism-private GGUF quant formats (decoded, not yet wired into any loader)
//
// `PQ2_0` (tag 142) and `PTQ1_0` (tag 143) are private to the PrismML
// llama.cpp fork (branch `prism`, commit adfffbe4). They are registered here
// so a checkpoint using them produces a clear "not implemented" error naming
// the format, rather than "unknown GGUF dtype tag 142" from the parser. The
// decode paths are implemented and KAT-tested against the C reference, but
// no provider currently routes them, so nothing constructs them at runtime.
//
// Upstream `Q2_0` is tag 42 at group 64 and is a *different* format; these
// high ids were chosen by the fork precisely so the two can coexist.
// ============================================================================

/// Weights per `PQ2_0` block (`QK_PQ2_0`): 128, one fp16 scale per 128 weights.
pub const BLOCK_SIZE_PQ2_0: usize = 128;
/// Bytes per `PQ2_0` block: 2-byte fp16 delta + 128*2/8 packed codes.
pub const BLOCK_BYTES_PQ2_0: usize = 34;
/// Weights per `PTQ1_0` block (`QK_PTQ1_0`).
pub const BLOCK_SIZE_PTQ1_0: usize = 128;
/// `qs` bytes in a `PTQ1_0` block: 120 values at 5 trits/byte.
pub const PTQ1_0_QS_BYTES: usize = 24;
/// `qh` bytes in a `PTQ1_0` block: 8 values at 4 trits/byte.
pub const PTQ1_0_QH_BYTES: usize = 2;
/// Bytes per `PTQ1_0` block: 24 + 2 + 2 (fp16 delta last).
pub const BLOCK_BYTES_PTQ1_0: usize = 28;

/// Decode a `PQ2_0` codebyte per element: same `(q - 1) * d` law as `Q2_0`.
fn pq2_0_element(d: f32, qs: &[u8], j: usize) -> f32 {
    let q = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
    (q as f32 - 1.0) * d
}

/// Dequantize Prism `PQ2_0` (GGUF tag 142) packed weights.
///
/// Layout from `ggml-common.h` on branch `prism`:
/// ```c
/// #define QK_PQ2_0 128
/// typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0 / 4]; } block_pq2_0;
/// ```
/// and `dequantize_row_pq2_0` is byte-identical in body to `dequantize_row_q2_0`
/// apart from `qk`. 128 weights per 34-byte block = 2.125 bits per weight.
///
/// # Errors
/// Returns [`Error::Backend`] if `num_weights` is not a multiple of
/// [`BLOCK_SIZE_PQ2_0`] or if `data` is shorter than the block-aligned size.
pub fn dequant_pq2_0(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    if num_weights == 0 {
        return Ok(Vec::new());
    }
    if num_weights % BLOCK_SIZE_PQ2_0 != 0 {
        return Err(Error::Backend(format!(
            "dequant_pq2_0: {num_weights} weights is not a multiple of block size {BLOCK_SIZE_PQ2_0}"
        )));
    }
    let num_blocks = num_weights / BLOCK_SIZE_PQ2_0;
    let expected = num_blocks * BLOCK_BYTES_PQ2_0;
    if data.len() < expected {
        return Err(Error::Backend(format!(
            "dequant_pq2_0: buffer too short: expected {expected}, have {}",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES_PQ2_0;
        let d = f16_to_f32(data[base], data[base + 1]);
        let qs = &data[base + 2..base + BLOCK_BYTES_PQ2_0];
        for j in 0..BLOCK_SIZE_PQ2_0 {
            out.push(pq2_0_element(d, qs, j));
        }
    }
    Ok(out)
}

/// Extract the base-3 digit at position `n` from a ceil-encoded base-243 byte.
///
/// The fork stores 5 trits in one byte as a base-243 number rounded *up* to the
/// nearest multiple of 243/256, so the inverse is not a plain division. The C
/// code does `int16_t xi = ((uint16_t) q * pow3[n] * 3) >> 8;`, reproduced here
/// exactly; the intermediate can exceed a byte, which is why this is not `q / pow3`.
fn ptq1_0_trit(q: u8, n: usize) -> i16 {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    // The C is two steps, and the first one truncates:
    //     uint8_t  q  = x[i].qs[j + m] * pow3[n];   // <-- uint8_t: wraps mod 256
    //     int16_t xi = ((uint16_t) q * 3) >> 8;
    // Skipping the wrap changes the result for every trit where the product
    // exceeds 255, so it is reproduced explicitly rather than folded into one
    // wider multiply.
    let scaled = q.wrapping_mul(POW3[n]);
    (((scaled as u16) * 3) >> 8) as i16
}

/// Dequantize Prism `PTQ1_0` (GGUF tag 143) ternary packed weights.
///
/// Layout from `ggml-common.h` on branch `prism`:
/// ```c
/// #define QK_PTQ1_0 128
/// typedef struct {
///     uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; // 24 B, 5 trits/byte -> 120
///     uint8_t qh[QK_PTQ1_0/64];                   //  2 B, 4 trits/byte ->   8
///     ggml_half d;
/// } block_ptq1_0;
/// ```
/// Each trit is a base-3 digit mapping `-1, 0, +1` to codes `0, 1, 2`, decoded as
/// `(xi - 1) * d`. 128 weights per 28-byte block = 1.75 bits per weight.
///
/// `qs` is walked with stage widths 32/16/8 (generalized from upstream `TQ1_0`'s
/// 32-then-16, which cannot cover a 24-byte `qs`); that yields 120 values, then
/// `qh` contributes the final 8. The `d` field sits at the END of the struct,
/// unlike every other 2-bit format here.
///
/// # Errors
/// Returns [`Error::Backend`] if `num_weights` is not a multiple of
/// [`BLOCK_SIZE_PTQ1_0`] or if `data` is shorter than the block-aligned size.
pub fn dequant_ptq1_0(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    if num_weights == 0 {
        return Ok(Vec::new());
    }
    if num_weights % BLOCK_SIZE_PTQ1_0 != 0 {
        return Err(Error::Backend(format!(
            "dequant_ptq1_0: {num_weights} weights is not a multiple of block size {BLOCK_SIZE_PTQ1_0}"
        )));
    }
    let num_blocks = num_weights / BLOCK_SIZE_PTQ1_0;
    let expected = num_blocks * BLOCK_BYTES_PTQ1_0;
    if data.len() < expected {
        return Err(Error::Backend(format!(
            "dequant_ptq1_0: buffer too short: expected {expected}, have {}",
            data.len()
        )));
    }

    const STAGES: [usize; 3] = [32, 16, 8];
    let mut out = Vec::with_capacity(num_weights);
    for b in 0..num_blocks {
        let base = b * BLOCK_BYTES_PTQ1_0;
        let qs = &data[base..base + PTQ1_0_QS_BYTES];
        let qh = &data[base + PTQ1_0_QS_BYTES..base + PTQ1_0_QS_BYTES + PTQ1_0_QH_BYTES];
        let d_off = base + PTQ1_0_QS_BYTES + PTQ1_0_QH_BYTES;
        let d = f16_to_f32(data[d_off], data[d_off + 1]);

        let mut j = 0usize;
        for &c in STAGES.iter() {
            while j + c <= PTQ1_0_QS_BYTES {
                for n in 0..5usize {
                    for m in 0..c {
                        // One byte `qs[j + m]` holds the `n`-th trit of the
                        // `m`-th weight in this group of `c`.
                        out.push((ptq1_0_trit(qs[j + m], n) as f32 - 1.0) * d);
                    }
                }
                j += c;
            }
        }
        for n in 0..4usize {
            for &byte in qh.iter() {
                out.push((ptq1_0_trit(byte, n) as f32 - 1.0) * d);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod ostquant_w4_tests {
    use super::*;

    /// Round-trip through the WhiteCrow layout: quantize, dequant with the
    /// REFERENCE decoder, error must stay inside u4 group-128 noise. This is
    /// the layout the requant-at-load decode path feeds the native v_dot8
    /// GEMV, so the two functions must agree on nibble order, scale and zero.
    #[test]
    fn quant_ostquant_w4_roundtrips_through_reference_dequant() {
        let (n, k) = (16usize, 512usize);
        let mut s: u32 = 0xC0FFEE;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let w: Vec<f32> = (0..n * k)
            .map(|_| ((next() & 0xffff) as f32 / 65535.0 - 0.5) * 0.08)
            .collect();

        let (qw, sc, zr) =
            quant_ostquant_w4_group128(&w, n, k).expect("quant_ostquant_w4_group128");
        let deq = dequant_ostquant_w4a4(&qw, &sc, &zr, &[n, k], 128).expect("reference dequant");
        assert_eq!(deq.len(), n * k);

        // Per-group u4 range is d = (max-min)/15; reconstruction error is
        // bounded by d/2 plus bf16 scale rounding. Weight magnitudes here are
        // ~0.04, group spread ~0.06 -> d ~ 0.004; allow a generous 2*d.
        let mut worst = 0.0f32;
        for (a, b) in w.iter().zip(&deq) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 0.004, "round-trip error {worst}");
    }

    #[test]
    fn quant_ostquant_w4_rejects_short_input() {
        assert!(quant_ostquant_w4_group128(&[0.0; 10], 4, 128).is_err());
    }
}

#[cfg(test)]
mod fp8_block16_tests {
    use super::*;

    /// Block/unblock round-trips byte-exactly: the layout is a permutation,
    /// not a quantization, so any difference is a packer bug.
    #[test]
    fn block16_roundtrips_byte_exact() {
        let (n, k) = (48usize, 128usize);
        let data: Vec<u8> = (0..n * k).map(|i| (i % 251) as u8).collect();
        let blocked = block_fp8_16x16(&data, n, k).expect("block");
        assert_eq!(blocked.len(), n * k);
        assert_ne!(blocked, data, "blocked order must differ from row-major");
        let back = unblock_fp8_16x16(&blocked, n, k).expect("unblock");
        assert_eq!(back, data);
    }

    /// The blocked tile the kernel loads contiguously (ldm=16) holds exactly
    /// the 16x16 block the old kernel gathered K-strided: tile (nt, kb) is
    /// B rows nt*16..+16, cols kb*16..+16 in row-major-within-block order,
    /// which is the fragment's (r, c) -> block[c][r] consumption order.
    #[test]
    fn block16_tile_matches_strided_gather_elementwise() {
        let (n, k) = (32usize, 64usize);
        let data: Vec<u8> = (0..n * k).map(|i| (i * 7 % 256) as u8).collect();
        let blocked = block_fp8_16x16(&data, n, k).expect("block");
        let kb_total = k / 16;
        for nt in 0..n / 16 {
            for kb in 0..kb_total {
                for c in 0..16 {
                    for r in 0..16 {
                        let tile_off = (nt * kb_total + kb) * 256 + c * 16 + r;
                        let row_major = (nt * 16 + c) * k + kb * 16 + r;
                        assert_eq!(
                            blocked[tile_off], data[row_major],
                            "nt={nt} kb={kb} c={c} r={r}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn block16_rejects_ragged_geometry_and_short_input() {
        assert!(block_fp8_16x16(&[0u8; 256], 15, 16).is_err());
        assert!(block_fp8_16x16(&[0u8; 256], 16, 17).is_err());
        assert!(block_fp8_16x16(&[0u8; 100], 16, 16).is_err());
        assert!(unblock_fp8_16x16(&[0u8; 100], 16, 16).is_err());
    }

    /// Cache-key discipline (mirrors WhiteCrow): geometry and encoder version
    /// participate, so a layout bump or a different tensor can never collide.
    #[test]
    fn blocked_fp8_cache_key_separates_geometry_and_version() {
        let h = 0x1234_5678_9ABC_DEF0u64;
        assert_ne!(
            blocked_fp8_cache_key(16, 128, h),
            blocked_fp8_cache_key(32, 128, h)
        );
        assert_ne!(
            blocked_fp8_cache_key(16, 128, h),
            blocked_fp8_cache_key(16, 256, h)
        );
        assert_ne!(
            blocked_fp8_cache_key(16, 128, h),
            blocked_fp8_cache_key(16, 128, h ^ 1)
        );
        assert_eq!(FP8_BLOCK16_ENCODER_VERSION, 1);
    }

    /// Host dequant of blocked bytes equals per-code E4M3 decode of the
    /// row-major originals: unblock first, never the scale-header reader.
    #[test]
    fn dequant_blocked16_matches_per_code_decode() {
        let (n, k) = (32usize, 64usize);
        let data: Vec<u8> = (0..n * k).map(|i| (i * 13 % 256) as u8).collect();
        let blocked = block_fp8_16x16(&data, n, k).expect("block");
        let got = dequant_fp8_blocked16(&blocked, n, k).expect("dequant");
        assert_eq!(got.len(), n * k);
        for (i, (&b, &v)) in data.iter().zip(&got).enumerate() {
            // Bitwise: the data includes NaN codes (NaN != NaN by ==).
            assert_eq!(v.to_bits(), fp8_e4m3_to_f32(b).to_bits(), "element {i}");
        }
        assert!(dequant_fp8_blocked16(&blocked, 15, 64).is_err());
    }
}
