//! Reference CPU-side dequantization algorithms for `.grim` formats.
//!
//! Provides `dequant_row` which matches the ROCm kernel behavior exactly,
//! verifying the numerics of mixed bitwidth, row scale, backup layer, and outlier merges.

use grim_format::spec::GrimTensorExt;

/// Dequantize a single row using the mixed-bitwidth, row scale, residual backup layers,
/// and outlier streams.
///
/// # Arguments
/// - `row_idx`: index of the row being dequantized.
/// - `row_stride`: length of the row.
/// - `packed_bits`: the raw bit-packed codes for the entire tensor.
/// - `scales`: the raw scales bytes.
/// - `default_bpw`: uniform bits per weight fallback.
/// - `ext`: Optional reference to the GrimTensorExt for advanced properties (e.g. backup layers).
/// - `outliers`: Decoded index-value outliers.
pub fn dequant_row(
    row_idx: usize,
    row_stride: usize,
    packed_bits: &[u8],
    scales: &[u8],
    default_bpw: u8,
    ext: Option<&GrimTensorExt>,
    outliers: &[(u32, f32)],
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_stride];
    let bpw = ext.map(|e| e.default_bpw).unwrap_or(default_bpw);
    
    let row_bytes = ((row_stride * bpw as usize + 7) / 8 + 255) & !255;
    let row_start_idx = row_idx * row_bytes;
    let row_data = if row_start_idx < packed_bits.len() {
        &packed_bits[row_start_idx..]
    } else {
        &[]
    };

    for i in 0..row_stride {
        let bit_offset = i * bpw as usize;
        let byte_offset = bit_offset / 8;
        let in_byte_offset = bit_offset % 8;
        let bits_left_in_byte = 8 - in_byte_offset;
        
        let code = if byte_offset < row_data.len() {
            if bits_left_in_byte >= bpw as usize {
                let shift = bits_left_in_byte - bpw as usize;
                ((row_data[byte_offset] >> shift) & ((1 << bpw) - 1)) as u32
            } else {
                let high_bits = bits_left_in_byte;
                let low_bits = bpw as usize - high_bits;
                let high_part = (row_data[byte_offset] & ((1 << high_bits) - 1)) as u32;
                let low_part = if byte_offset + 1 < row_data.len() {
                    let shift = 8 - low_bits;
                    ((row_data[byte_offset + 1] >> shift) & ((1 << low_bits) - 1)) as u32
                } else {
                    0
                };
                (high_part << low_bits) | low_part
            }
        } else {
            0
        };

        let levels = (1u32 << bpw) as f32;
        let normalized = code as f32 / (levels - 1.0);
        out[i] = normalized * 2.0 - 1.0;
    }

    let scale_val = if !scales.is_empty() && row_idx < scales.len() {
        scales[row_idx] as f32 / 255.0f32
    } else {
        1.0f32
    };

    for val in out.iter_mut() {
        *val *= scale_val;
    }

    if let Some(ext) = ext {
        if ext.backup1.is_present() && ext.gptq_ordered > 0 {
            let b1_bpw = ext.backup1.bpw;
            let b1_row_bytes = ((row_stride * b1_bpw as usize + 7) / 8 + 255) & !255;
            let b1_row_start = ext.backup1.codes_offset as usize + row_idx * b1_row_bytes;
            
            let b1_scale_idx = ext.backup1.scale_offset as usize + row_idx;
            let b1_scale = if b1_scale_idx < packed_bits.len() {
                packed_bits[b1_scale_idx] as f32 / 255.0f32
            } else {
                1.0f32
            };

            for i in 0..row_stride {
                let bit_offset = i * b1_bpw as usize;
                let byte_offset = bit_offset / 8;
                let in_byte_offset = bit_offset % 8;
                let bits_left_in_byte = 8 - in_byte_offset;
                
                let b1_row_data = if b1_row_start < packed_bits.len() {
                    &packed_bits[b1_row_start..]
                } else {
                    &[]
                };

                let code = if byte_offset < b1_row_data.len() {
                    if bits_left_in_byte >= b1_bpw as usize {
                        let shift = bits_left_in_byte - b1_bpw as usize;
                        ((b1_row_data[byte_offset] >> shift) & ((1 << b1_bpw) - 1)) as u32
                    } else {
                        let high_bits = bits_left_in_byte;
                        let low_bits = b1_bpw as usize - high_bits;
                        let high_part = (b1_row_data[byte_offset] & ((1 << high_bits) - 1)) as u32;
                        let low_part = if byte_offset + 1 < b1_row_data.len() {
                            let shift = 8 - low_bits;
                            ((b1_row_data[byte_offset + 1] >> shift) & ((1 << low_bits) - 1)) as u32
                        } else {
                            0
                        };
                        (high_part << low_bits) | low_part
                    }
                } else {
                    0
                };

                let levels = (1u32 << b1_bpw) as f32;
                let normalized = code as f32 / (levels - 1.0);
                let dequant_b1 = (normalized * 2.0 - 1.0) * b1_scale;
                out[i] += dequant_b1;
            }
        }
    }

    for &(idx, val) in outliers {
        let r = idx as usize / row_stride;
        let c = idx as usize % row_stride;
        if r == row_idx && c < out.len() {
            out[c] = val;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple packing helper local to this test block to avoid dependencies on grim-backend-rocm.
    fn pack_row_test(weights: &[f32], bpw: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let bits_needed = weights.len() as u64 * bpw as u64;
        let bytes_needed = (bits_needed + 7) / 8;
        out.resize(bytes_needed as usize, 0u8);
        
        for (i, &v) in weights.iter().enumerate() {
            let levels = (1u32 << bpw) as f32;
            let normalized = (v.clamp(-1.0, 1.0) + 1.0) * 0.5;
            let code = (normalized * (levels - 1.0)).round() as u32;
            
            let bit_offset = i * bpw as usize;
            let byte_offset = bit_offset / 8;
            let in_byte_offset = bit_offset % 8;
            let bits_left_in_byte = 8 - in_byte_offset;
            
            if bits_left_in_byte >= bpw as usize {
                let shift = bits_left_in_byte - bpw as usize;
                out[byte_offset] |= (code << shift) as u8;
            } else {
                let high_bits = bits_left_in_byte;
                let low_bits = bpw as usize - high_bits;
                out[byte_offset] |= (code >> low_bits) as u8;
                if byte_offset + 1 < bytes_needed as usize {
                    let low_shift = 8 - low_bits;
                    out[byte_offset + 1] |= (code << low_shift) as u8;
                }
            }
        }
        
        let aligned = (out.len() + 255) & !255;
        out.resize(aligned, 0u8);
        out
    }

    #[test]
    fn test_dequant_row_basic() {
        let weights = vec![0.5f32, -0.2f32, 0.8f32, -1.0f32, 1.0f32, 0.0f32, 0.3f32, -0.7f32];
        let bpw = 4;
        
        let packed = pack_row_test(&weights, bpw);

        let scales = vec![255u8];
        let outliers = vec![(2u32, 999.0f32)];
        
        let dequantized = dequant_row(0, 8, &packed, &scales, bpw, None, &outliers);
        
        assert_eq!(dequantized.len(), 8);
        assert_eq!(dequantized[2], 999.0f32);
        
        let max_err = 1.0 / 15.0f32 + 1e-5;
        for i in [0, 1, 3, 4, 5, 6, 7] {
            assert!((dequantized[i] - weights[i]).abs() <= max_err);
        }
    }
}
