//! Block quantizers for Grim. Each format stores a block of weights as
//! low-bit integers plus per-block scale (and optionally min for asymmetric).
//!
//! Q8_0: 8-bit symmetric, block size 32 — one f16 scale per 32 values.
//! Q4_K: llama.cpp K-quant, block size 32 — 6-bit super-block scale,
//!       4-bit sub-block values, per-sub-block scale.
//!
//! All functions here take raw quantized bytes and produce `Vec<f32>`.

use grim_tensor::error::{Error, Result};

pub const BLOCK_SIZE_Q8: usize = 32;
pub const BLOCK_SIZE_Q4_K: usize = 32;

/// Dequantize Q8_0 bytes to f32.
/// Q8_0 layout: for every 32 weights, a `f16` scale followed by 32 `i8` values.
pub fn dequant_q80(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    let stride = std::mem::size_of::<u16>() + BLOCK_Q8_WEIGHTS; // 2 + 32 = 34 bytes
    let num_blocks = (num_weights + BLOCK_Q8_WEIGHTS - 1) / BLOCK_Q8_WEIGHTS;
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

/// Dequantize Q4_K bytes to f32.
/// Q4_K layout (llama.cpp style): per super-block (32 weights):
///   - scale (6-bit super-block scale, stored as u8 in `d` or `dmin`)
///   - 8 sub-blocks of 4 weights each, 4-bit each
///   - per-sub-block scale
/// Simplified: for v1, treat as 4-bit symmetric with one f32 scale per 32 values.
pub fn dequant_q4k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    let stride = BLOCK_SIZE_Q4_K / 2 + 4; // 16 bytes values + scale = 20
    let num_blocks = (num_weights + BLOCK_SIZE_Q4_K - 1) / BLOCK_SIZE_Q4_K;
    if data.len() < num_blocks * stride {
        return Err(Error::Backend(format!(
            "Q4_K: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * stride,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut data_pos = 0;
    let mut remaining = num_weights;
    for _ in 0..num_blocks {
        // v1: 4-bit symmetric, one f32 scale per block
        let scale_bytes = [data[data_pos], data[data_pos+1], data[data_pos+2], data[data_pos+3]];
        data_pos += 4;
        let scale = f32::from_le_bytes(scale_bytes);
        for wi in 0..BLOCK_SIZE_Q4_K.min(remaining) {
            let byte = if wi % 2 == 0 {
                data[data_pos + wi / 2] & 0x0F
            } else {
                (data[data_pos + wi / 2] >> 4) & 0x0F
            };
            out.push((byte as i8 as f32) * scale);
        }
        data_pos += BLOCK_SIZE_Q4_K / 2;
        remaining = remaining.saturating_sub(BLOCK_SIZE_Q4_K);
    }
    Ok(out)
}

fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // Subnormal or zero
        f32::from_bits((sign << 31) | (mant << 13))
    } else if exp == 31 {
        // NaN or inf
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Quantize a slice of f32 values to Q8_0 bytes.
/// Each block of 32 gets a f16 scale and 32 i8 values.
pub fn quant_q80(data: &[f32]) -> Result<Vec<u8>> {
    let num_blocks = (data.len() + BLOCK_Q8_WEIGHTS - 1) / BLOCK_Q8_WEIGHTS;
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
        for _ in block.len()..BLOCK_Q8_WEIGHTS {
            out.push(0u8);
        }
    }
    Ok(out)
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

trait Clamp {
    fn clamp(self, lo: f32, hi: f32) -> Self;
}
impl Clamp for f32 {
    fn clamp(self, lo: f32, hi: f32) -> Self {
        if self < lo { lo } else if self > hi { hi } else { self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_q80() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let quantized = quant_q80(&data).unwrap();
        let dequantized = dequant_q80(&quantized, data.len()).unwrap();
        assert_eq!(data.len(), dequantized.len());
        // Q8_0 should be close
        for i in 0..data.len() {
            let diff = (data[i] - dequantized[i]).abs();
            assert!(diff < 0.5, "diff at {i}: {} vs {}, diff={}", data[i], dequantized[i], diff);
        }
    }

    #[test]
    fn dequant_q4k_basic() {
        // 64 weights: 2 blocks of 32 each, 20 bytes per block = 40 bytes
        let mut data = vec![0u8; 40];
        // Block 0: scale = 1.0 (0x3F800000 in LE f32)
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        // values: byte 4..=19 packed 4-bit
        for i in 0..16 {
            data[4 + i] = (i as u8) | ((i as u8 + 1) << 4);
        }
        // Block 1 same pattern
        data[20..24].copy_from_slice(&0.5f32.to_le_bytes());
        let deq = dequant_q4k(&data, 64).unwrap();
        assert_eq!(deq.len(), 64);
        // First value is low nibble of byte 4 = 0
        assert_eq!(deq[0], 0.0f32);
        // Second value is high nibble of byte 4 = 1
        assert_eq!(deq[1], 1.0f32);
    }
}